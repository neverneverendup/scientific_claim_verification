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
from util import arg2param, flatten, stance2json, merge_json
from my_models import StanceParagraphClassifier as JointParagraphClassifier
# 注意这个作者写代码，这竟然用别名，容易造成误导！ stance prediction 模型的训练使用了负采样
from dataset import SciFactStanceDataset, SciFactStancePredictionDataset

import logging

def reset_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def predict(model, dataset):
    model.eval()
    stance_preds = []
    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size = args.eval_batch_size, shuffle=False)):
            encoded_dict = encode(tokenizer, batch)
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], 
                                                           tokenizer.sep_token_id, args.repfile)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            stance_out, _ = model(encoded_dict, transformation_indices)
            stance_preds.extend(stance_out)                

    return stance_preds

def complete_no_rationale_samples_preds(stance_jsonl, rationale_json):
    completed_stance_data = []
    for stance_pred, rationale_pred in zip(stance_json, rationale_json):
        if stance_pred["claim_id"] == rationale_pred["claim_id"]:
            completed_stance_data.append(stance_pred)
        else:
            completed_stance_data.append({})
            stance_pred = {"claim_id": rationale_pred["claim_id"], "labels":{}}
            for doc_id, pred in rationale_pred["evidence"].items():
                pass
            for doc_id, pred in rationale_pred["evidence"].items():
                if len(pred) == 0:
                    stance_pred["labels"][doc_id]["label"] = "NOT_ENOUGH_INFO"
    return stance_json

def evaluation(model, dataset, dummy=True):
    model.eval()
    stance_preds = []
    stance_labels = []

    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size = args.eval_batch_size, shuffle=False)):
            encoded_dict = encode(tokenizer, batch)
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], 
                                                               tokenizer.sep_token_id, args.repfile)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            stance_label = batch["stance"].to(device)
            stance_out, loss = \
                model(encoded_dict, transformation_indices, stance_label = stance_label)
            stance_preds.extend(stance_out)
            stance_labels.extend(stance_label.cpu().numpy().tolist())

    stance_f1 = f1_score(stance_labels,stance_preds,average="micro",labels=[1, 2])
    stance_precision = precision_score(stance_labels,stance_preds,average="micro",labels=[1, 2])
    stance_recall = recall_score(stance_labels,stance_preds,average="micro",labels=[1, 2])
    return stance_f1, stance_precision, stance_recall



def encode(tokenizer, batch, max_sent_len = 512):
    def truncate(input_ids, max_length, sep_token_id, pad_token_id):
        def longest_first_truncation(sentences, objective):
            sent_lens = [len(sent) for sent in sentences]
            while np.sum(sent_lens) > objective:
                max_position = np.argmax(sent_lens)
                sent_lens[max_position] -= 1
            return [sentence[:length] for sentence, length in zip(sentences, sent_lens)]

        all_paragraphs = []
        for paragraph in input_ids:
            valid_paragraph = paragraph[paragraph != pad_token_id]
            if valid_paragraph.size(0) <= max_length:
                all_paragraphs.append(paragraph[:max_length].unsqueeze(0))
            else:
                sep_token_idx = np.arange(valid_paragraph.size(0))[(valid_paragraph == sep_token_id).numpy()]
                idx_by_sentence = []
                prev_idx = 0
                for idx in sep_token_idx:
                    idx_by_sentence.append(paragraph[prev_idx:idx])
                    prev_idx = idx
                objective = max_length - 1 - len(idx_by_sentence[0]) # The last sep_token left out
                truncated_sentences = longest_first_truncation(idx_by_sentence[1:], objective)
                truncated_paragraph = torch.cat([idx_by_sentence[0]] + truncated_sentences + [torch.tensor([sep_token_id])],0)
                all_paragraphs.append(truncated_paragraph.unsqueeze(0))

        return torch.cat(all_paragraphs, 0)

    inputs = zip(batch["claim"], batch["paragraph"])
    encoded_dict = tokenizer.batch_encode_plus(
        inputs,
        pad_to_max_length=True,add_special_tokens=True,
        return_tensors='pt')
    if encoded_dict['input_ids'].size(1) > max_sent_len:
        if 'token_type_ids' in encoded_dict:
            encoded_dict = {
                "input_ids": truncate(encoded_dict['input_ids'], max_sent_len, 
                                      tokenizer.sep_token_id, tokenizer.pad_token_id),
                'token_type_ids': encoded_dict['token_type_ids'][:,:max_sent_len],
                'attention_mask': encoded_dict['attention_mask'][:,:max_sent_len]
            }
        else:
            encoded_dict = {
                "input_ids": truncate(encoded_dict['input_ids'], max_sent_len, 
                                      tokenizer.sep_token_id, tokenizer.pad_token_id),
                'attention_mask': encoded_dict['attention_mask'][:,:max_sent_len]
            }

    return encoded_dict

def post_process_stance(rationale_json, stance_json):
    print(len(rationale_json) ,len(stance_json))
    assert(len(rationale_json) == len(stance_json))
    for stance_pred, rationale_pred in zip(stance_json, rationale_json):
        assert(stance_pred["claim_id"] == rationale_pred["claim_id"])
        for doc_id, pred in rationale_pred["evidence"].items():
            if len(pred) == 0:
                if doc_id not in stance_pred["labels"].keys():
                    stance_pred["labels"][doc_id] = {}

                stance_pred["labels"][doc_id]["label"] = "NOT_ENOUGH_INFO"
    return stance_json

def sent_rep_indices(input_ids, sep_token_id, model_name):

    """
    Compute the [SEP] indices matrix of the BERT output.
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
    Compute the token indices matrix of the BERT output.
    input_ids: (batch_size, paragraph_len)
    batch_indices, indices_by_batch, mask: (batch_size, N_sentence, N_token)
    bert_out: (batch_size, paragraph_len,BERT_dim)
    bert_out[batch_indices,indices_by_batch,:]: (batch_size, N_sentence, N_token, BERT_dim)
    """
    padding_idx = -1
    sep_tokens = (input_ids == sep_token_id).bool()
    paragraph_lens = torch.sum(sep_tokens,1).numpy().tolist()
    indices = torch.arange(sep_tokens.size(-1)).unsqueeze(0).expand(sep_tokens.size(0),-1)
    sep_indices = torch.split(indices[sep_tokens],paragraph_lens)
    paragraph_lens = []
    all_word_indices = []
    for paragraph in sep_indices:
        if "large" in model_name:
            paragraph = paragraph[1:]
        word_indices = [torch.arange(paragraph[i]+1, paragraph[i+1]+1) for i in range(paragraph.size(0)-1)]
        paragraph_lens.append(len(word_indices))
        all_word_indices.extend(word_indices)
    indices_by_sentence = nn.utils.rnn.pad_sequence(all_word_indices, batch_first=True, padding_value=padding_idx)
    indices_by_sentence_split = torch.split(indices_by_sentence,paragraph_lens)
    indices_by_batch = nn.utils.rnn.pad_sequence(indices_by_sentence_split, batch_first=True, padding_value=padding_idx)
    batch_indices = torch.arange(sep_tokens.size(0)).unsqueeze(-1).unsqueeze(-1).expand(-1,indices_by_batch.size(1),indices_by_batch.size(-1))
    mask = (indices_by_batch>=0) 

    return batch_indices.long(), indices_by_batch.long(), mask.long()

def load_jsonl(file):
    return [rationale for rationale in jsonlines.open(file)]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--repfile', type=str, default = "roberta-large", help="Word embedding file")
    argparser.add_argument('--corpus_file', type=str, default="/nas/home/xiangcil/para_scifact/data/corpus.jsonl")
    argparser.add_argument('--train_file', type=str)
    argparser.add_argument('--test_file', type=str, default="/nas/home/xiangcil/CitationEvaluation/SciFact/claims_dev_biosent_retrieved.jsonl")
    argparser.add_argument('--bert_lr', type=float, default=5e-6, help="Learning rate for BERT-like LM")
    argparser.add_argument('--lr', type=float, default=1e-6, help="Learning rate")
    argparser.add_argument('--dropout', type=float, default=0, help="embedding_dropout rate")
    argparser.add_argument('--bert_dim', type=int, default=1024, help="bert_dimension")
    argparser.add_argument('--epoch', type=int, default=20, help="Training epoch")
    argparser.add_argument('--max_seq_length', type=int, default=512)
    argparser.add_argument('--checkpoint', type=str, default = "scifact_roberta_stance_paragraph.model")
    argparser.add_argument('--log_file', type=str, default = "stance_paragraph_roberta_performances.jsonl")
    argparser.add_argument('--prediction', type=str, default = "prediction_scifact_roberta_stance_paragraph.jsonl")
    argparser.add_argument('--update_step', type=int, default=10)

    argparser.add_argument('--train_batch_size', type=int, default=1) # roberta-large: 2; bert: 8
    argparser.add_argument('--eval_batch_size', type=int, default=25) # roberta-large: 2; bert: 8

    argparser.add_argument('--k', type=int, default=0)
    argparser.add_argument('--output_dir', type=str, default="model_dir/label_prediction.jsonl")

    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    reset_random_seed(12345)
    args = argparser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.repfile)

    if args.test_file:
        test = True
    else:
        test = False

    params = vars(args)

    for k,v in params.items():
        print(k,v)

    rationale_json_file = os.path.join(args.output_dir, 'rationale_selection.jsonl')
    dev_set = SciFactStancePredictionDataset(args.corpus_file, args.test_file, rationale_json_file)
    model = JointParagraphClassifier(args.repfile, args.bert_dim, 
                                      args.dropout)#.to(device)

    model.load_state_dict(torch.load(args.checkpoint))
    print("Loaded saved model.")
    model = model.to(device)

    stance_preds = predict(model, dev_set)
    stance_json = stance2json(dev_set.samples, stance_preds, dev_set.excluded_pairs)
    rationale_json = load_jsonl(rationale_json_file)
    stance_json = post_process_stance(rationale_json, stance_json)

    if args.output_dir is not None:
        with jsonlines.open(os.path.join(args.output_dir, 'label_prediction.jsonl'), 'w') as output:
            for result in stance_json:
                output.write(result)