import argparse

import torch
import jsonlines
import os

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score

import random
import numpy as np

from tqdm import tqdm
from util import arg2param, flatten, rationale2json
from paragraph_model_dynamic import RationaleParagraphClassifier as JointParagraphClassifier
from dataset import SciFactParagraphBatchDataset

import logging

def reset_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def batch_rationale_label(labels, padding_idx = 2):
    max_sent_len = max([len(label) for label in labels])
    label_matrix = torch.ones(len(labels), max_sent_len) * padding_idx
    label_list = []
    for i, label in enumerate(labels):
        for j, evid in enumerate(label):
            label_matrix[i,j] = int(evid)
        label_list.append([int(evid) for evid in label])
    return label_matrix.long(), label_list

def predict(model, dataset):
    model.eval()
    rationale_predictions = []

    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size = args.batch_size, shuffle=False)):
            encoded_dict = encode(tokenizer, batch)
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], 
                                                           tokenizer.sep_token_id, args.repfile)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            rationale_out, _= model(encoded_dict, transformation_indices)
            rationale_predictions.extend(rationale_out)

    return rationale_predictions

def evaluation(model, dataset):
    model.eval()
    rationale_predictions = []
    rationale_labels = []
    sens_longer_than_max_len = 0
    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size = args.batch_size*5, shuffle=False)):
            count, encoded_dict = encode(tokenizer, batch)
            sens_longer_than_max_len += count
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], 
                                                               tokenizer.sep_token_id, args.repfile)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            padded_rationale_label, rationale_label = batch_rationale_label(batch["label"], padding_idx = 2)
            rationale_out, rationale_loss = \
                model(encoded_dict, transformation_indices, 
                      rationale_label = padded_rationale_label.to(device))

            rationale_predictions.extend(rationale_out)
            rationale_labels.extend(rationale_label)
            break

    print(sens_longer_than_max_len)
    print(rationale_predictions)

    rationale_f1 = f1_score(flatten(rationale_labels),flatten(rationale_predictions))
    rationale_precision = precision_score(flatten(rationale_labels),flatten(rationale_predictions))
    rationale_recall = recall_score(flatten(rationale_labels),flatten(rationale_predictions))
    return rationale_f1, rationale_precision, rationale_recall, rationale_predictions

def encode(tokenizer, batch, max_sent_len = 512):

    def truncate(input_ids, max_length, sep_token_id, pad_token_id):
        paragraph_length_longer_than_max = 0

        def longest_first_truncation(sentences, objective):
            sent_lens = [len(sent) for sent in sentences]
            while np.sum(sent_lens) > objective:
                max_position = np.argmax(sent_lens)
                sent_lens[max_position] -= 1
            return [sentence[:length] for sentence, length in zip(sentences, sent_lens)]

        all_paragraphs = []
        for paragraph in input_ids:
            print('paragraph',paragraph.shape)
            valid_paragraph = paragraph[paragraph != pad_token_id]
            print('valid',valid_paragraph.shape)
            if valid_paragraph.size(0) <= max_length:
                # tensor(512) -> tensor(1,512)
                print(paragraph[:max_length].shape,paragraph[:max_length].unsqueeze(0).shape)
                #print(paragraph[:max_length],paragraph[:max_length].unsqueeze(0))

                all_paragraphs.append(paragraph[:max_length].unsqueeze(0))
            else:
                paragraph_length_longer_than_max += 1
                sep_token_idx = np.arange(valid_paragraph.size(0))[(valid_paragraph == sep_token_id).numpy()]
                idx_by_sentence = []
                prev_idx = 0
                for idx in sep_token_idx:
                    idx_by_sentence.append(paragraph[prev_idx:idx])
                    prev_idx = idx
                objective = max_length - 1 - len(idx_by_sentence[0]) # The last sep_token left out
                # 总是截句子列表当前最长的句子
                truncated_sentences = longest_first_truncation(idx_by_sentence[1:], objective)
                #print('truncated_sentences', truncated_sentences.shape)
                truncated_paragraph = torch.cat([idx_by_sentence[0]] + truncated_sentences + [torch.tensor([sep_token_id])],0)
                print('truncated_paragraph', truncated_paragraph.shape)

                all_paragraphs.append(truncated_paragraph.unsqueeze(0))
        #print('all_paragraphs', all_paragraphs.shape)
        print('all_paragraphs', torch.cat(all_paragraphs, 0).shape) #(10, 512)
        print('超过512的篇章数量',paragraph_length_longer_than_max)
        return paragraph_length_longer_than_max, torch.cat(all_paragraphs, 0)

    inputs = zip(batch["claim"], batch["paragraph"])
    encoded_dict = tokenizer.batch_encode_plus(
        inputs,
        pad_to_max_length=True,add_special_tokens=True,
        return_tensors='pt')
    if encoded_dict['input_ids'].size(1) > max_sent_len:
        count, inputs = truncate(encoded_dict['input_ids'], max_sent_len,
                 tokenizer.sep_token_id, tokenizer.pad_token_id)
        if 'token_type_ids' in encoded_dict:
            encoded_dict = { # 若当前batch中的paragraph最大长度大于512，则截取保留512个，去除截取当时最长句子的部分token
                "input_ids": inputs ,
                'token_type_ids': encoded_dict['token_type_ids'][:,:max_sent_len],
                'attention_mask': encoded_dict['attention_mask'][:,:max_sent_len]
            }
        else:
            encoded_dict = {
                "input_ids": inputs ,
                'attention_mask': encoded_dict['attention_mask'][:,:max_sent_len]
            }

        return count, encoded_dict
    else:
        return 0, encoded_dict

def sent_rep_indices(input_ids, sep_token_id, model_name):

    """
    Compute the [SEP] indices matrix of the BERT outputs.
    input_ids: (batch_size, paragraph_len)
    batch_indices, indices_by_batch, mask: (batch_size, N_sentence)
    bert_out: (batch_size, paragraph_len,BERT_dim)
    bert_out[batch_indices,indices_by_batch,:]: (batch_size, N_sentence, BERT_dim)
    """

    sep_tokens = (input_ids == sep_token_id).bool()
    paragraph_lens = torch.sum(sep_tokens,1).numpy().tolist()
    indices = torch.arange(sep_tokens.size(-1)).unsqueeze(0).expand(sep_tokens.size(0),-1)
    sep_indices = torch.split(indices[sep_tokens],paragraph_lens)
    padded_sep_indices = nn.utils.rnn.pad_sequence(sep_indices, batch_first=True, padding_value=-1)
    batch_indices = torch.arange(padded_sep_indices.size(0)).unsqueeze(-1).expand(-1,padded_sep_indices.size(-1))
    mask = (padded_sep_indices>=0).long()

    if "roberta" in model_name:
        return batch_indices[:,2:], padded_sep_indices[:,2:], mask[:,2:]
    else:
        return batch_indices[:,1:], padded_sep_indices[:,1:], mask[:,1:]

def token_idx_by_sentence(input_ids, sep_token_id, model_name):
    """
    Compute the token indices matrix of the BERT outputs.
    input_ids: (batch_size, paragraph_len)
    batch_indices, indices_by_batch, mask: (batch_size, N_sentence, N_token)
    bert_out: (batch_size, paragraph_len,BERT_dim)
    bert_out[batch_indices,indices_by_batch,:]: (batch_size, N_sentence, N_token, BERT_dim)
    """
    padding_idx = -1
    sep_tokens = (input_ids == sep_token_id).bool()
    print('sep_tokens', sep_tokens)
    paragraph_lens = torch.sum(sep_tokens,1).numpy().tolist()
    print('paragraph_lens', paragraph_lens)
    indices = torch.arange(sep_tokens.size(-1)).unsqueeze(0).expand(sep_tokens.size(0),-1)
    print(indices)
    sep_indices = torch.split(indices[sep_tokens],paragraph_lens)
    print(indices[sep_tokens])
    print('sep_indices', sep_indices) #
    paragraph_lens = []
    all_word_indices = []
    for paragraph in sep_indices:
        if "large" in model_name:
            paragraph = paragraph[1:]
        word_indices = [torch.arange(paragraph[i]+1, paragraph[i+1]+1) for i in range(paragraph.size(0)-1)]
        print('word_indices', word_indices)
        paragraph_lens.append(len(word_indices))
        all_word_indices.extend(word_indices)
    print('all_word_indices', all_word_indices)
    indices_by_sentence = nn.utils.rnn.pad_sequence(all_word_indices, batch_first=True, padding_value=padding_idx)
    print('indices_by_sentence', indices_by_sentence, indices_by_sentence.shape)
    indices_by_sentence_split = torch.split(indices_by_sentence,paragraph_lens)
    print('indices_by_sentence_split', indices_by_sentence_split)
    indices_by_batch = nn.utils.rnn.pad_sequence(indices_by_sentence_split, batch_first=True, padding_value=padding_idx)
    print('indices_by_batch', indices_by_batch, indices_by_batch.shape)

    batch_indices = torch.arange(sep_tokens.size(0)).unsqueeze(-1).unsqueeze(-1).expand(-1,indices_by_batch.size(1),indices_by_batch.size(-1))
    mask = (indices_by_batch>=0) 
    print(batch_indices, batch_indices.shape)

    return batch_indices.long(), indices_by_batch.long(), mask.long()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--repfile', type=str, default = "roberta-large", help="Word embedding file")
    argparser.add_argument('--corpus_file', type=str, default="./data/para_scifact/corpus.jsonl")
    #argparser.add_argument('--train_file', type=str, default="./data/scifact/claims_train_retrieved.jsonl")
    argparser.add_argument('--train_file', type=str)
    argparser.add_argument('--pre_trained_model', type=str)
    argparser.add_argument('--test_file', type=str, default="./data/para_scifact/claims_dev_tfidf_retrieved.jsonl")
    argparser.add_argument('--bert_lr', type=float, default=1e-5, help="Learning rate for BERT-like LM")
    argparser.add_argument('--lr', type=float, default=5e-6, help="Learning rate")
    argparser.add_argument('--dropout', type=float, default=0, help="embedding_dropout rate")
    argparser.add_argument('--bert_dim', type=int, default=1024, help="bert_dimension")
    argparser.add_argument('--epoch', type=int, default=20, help="Training epoch")
    argparser.add_argument('--MAX_SENT_LEN', type=int, default=512)
    argparser.add_argument('--checkpoint', type=str, default = "./myargmodel/scifact_roberta_rationale_paragraph.model")
    argparser.add_argument('--log_file', type=str, default = "./myargmodel/test-rationale_paragraph_roberta_performances.jsonl")
    argparser.add_argument('--prediction', type=str, default = "./myargmodel/test-prediction_scifact_roberta_rationale_paragraph.jsonl")
    argparser.add_argument('--update_step', type=int, default=10)
    argparser.add_argument('--batch_size', type=int, default=2) # roberta-large: 2; bert: 8
    argparser.add_argument('--k', type=int, default=0)
    
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    
    reset_random_seed(12345)

    args = argparser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.repfile)

    if args.train_file:
        train = True
        #assert args.repfile is not None, "Word embedding file required for training."
        print('training mode...')
    else:
        train = False
    if args.test_file:
        test = True
        print('testing mode...')
    else:
        test = False

    params = vars(args)

    for k,v in params.items():
        print(k,v)

    if train:
        train_set = SciFactParagraphBatchDataset(args.corpus_file, args.train_file, 
                                                 sep_token = tokenizer.sep_token, k = args.k, dummy=False)
    dev_set = SciFactParagraphBatchDataset(args.corpus_file, args.test_file, 
                                           sep_token = tokenizer.sep_token, k = args.k, dummy=False)
    print('dev set ready... ')
    model = JointParagraphClassifier(args.repfile, args.bert_dim,
                                      args.dropout)#.to(device)

    if args.pre_trained_model is not None:
        print('load model')
        model.load_state_dict(torch.load(args.pre_trained_model))
        model.reinitialize()    ############
    
    model = model.to(device)
    
    if train:
        settings = [{'params': model.bert.parameters(), 'lr': args.bert_lr}]
        for module in model.extra_modules:
            settings.append({'params': module.parameters(), 'lr': args.lr})
        optimizer = torch.optim.Adam(settings)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epoch)
        model.train()

        prev_performance = 0
        for epoch in range(args.epoch):
            tq = tqdm(DataLoader(train_set, batch_size = args.batch_size, shuffle=True))
            for i, batch in enumerate(tq):
                encoded_dict = encode(tokenizer, batch)
                transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id, args.repfile)
                encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
                transformation_indices = [tensor.to(device) for tensor in transformation_indices]
                padded_rationale_label, rationale_label = batch_rationale_label(batch["label"], padding_idx = 2)
                rationale_out, loss = \
                    model(encoded_dict, transformation_indices, 
                          rationale_label = padded_rationale_label.to(device))
                loss.backward()

                if i % args.update_step == args.update_step - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    tq.set_description(f'Epoch {epoch}, iter {i}, loss: {round(loss.item(), 4)}')
            scheduler.step()

            # Evaluation
            train_score = evaluation(model, train_set)
            print(f'Epoch {epoch}, train rationale f1 p r: %.4f, %.4f, %.4f' % train_score)
            
            dev_score = evaluation(model, dev_set)
            print(f'Epoch {epoch}, rationale f1 p r: %.4f, %.4f, %.4f' % dev_score)

            dev_perf = dev_score[0]
            if dev_perf >= prev_performance:
                torch.save(model.state_dict(), args.checkpoint)
                best_state_dict = model.state_dict()
                prev_performance = dev_perf
                print("New model saved!")
            else:
                print("Skip saving model.")
        

    if test:
        if train:
            del model
            model = JointParagraphClassifier(args.repfile, args.bert_dim, 
                                              args.dropout).to(device)
            model.load_state_dict(best_state_dict)
            print("Testing on the new model.")
        else:
            model.load_state_dict(torch.load(args.checkpoint))
            print("Loaded saved model.")

        # Evaluation
        dev_score = evaluation(model, dev_set)
        print(f'Test rationale f1 p r: %.4f, %.4f, %.4f' % dev_score[:-1])
        
        params["rationale_f1"] = dev_score[0]
        params["rationale_precision"] = dev_score[1]
        params["rationale_recall"] = dev_score[2]

        with jsonlines.open(args.log_file, mode='a') as writer:
            writer.write(params)

        rationale_json = rationale2json(dev_set.samples, dev_score[-1])
        print(rationale_json)

        with jsonlines.open(args.prediction, mode='a') as writer:
            writer.write(rationale_json)