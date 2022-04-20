import argparse

import torch
import jsonlines
import os

import time
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
from my_models import RationaleParagraphClassifier

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
        for batch in tqdm(DataLoader(dataset, batch_size = args.eval_batch_size, shuffle=False)):
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

    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size = args.eval_batch_size, shuffle=False)):
            doc_ids, roles = get_batch_arg_feature(batch, evi_role)
            #arg_feature = pack_arg_feature_tensor(roles)
            #arg_feature = torch.tensor(arg_feature, dtype=torch.float32).to(device)
            #context_arg_feature = torch.tensor(context_arg_feature, dtype=torch.float32).to(device)

            encoded_dict = encode(tokenizer, batch)
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
    
    #print(rationale_predictions)

    rationale_f1 = f1_score(flatten(rationale_labels),flatten(rationale_predictions))
    rationale_precision = precision_score(flatten(rationale_labels),flatten(rationale_predictions))
    rationale_recall = recall_score(flatten(rationale_labels),flatten(rationale_predictions))
    return rationale_f1, rationale_precision, rationale_recall, rationale_predictions

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

'''
{'dataset': tensor([1, 1]), 'claim': ['Anthrax spores are very difficult to dispose once they are dispersed.', 'The mean suicide rate in women is lower after miscarriage than live birth.'], 'claim_id': tensor([ 114, 1228]), 'doc_id': tensor([33872649, 25641414]), 'paragraph': ["CONTEXT Bioterrorist attacks involving letters and mail-handling systems in Washington, DC, resulted in Bacillus anthracis (anthrax) spore contamination in the Hart Senate Office Building and other facilities in the US Capitol's vicinity. </s> OBJECTIVE To provide information about the nature and extent of indoor secondary aerosolization of B anthracis spores. </s> DESIGN Stationary and personal air samples, surface dust, and swab samples were collected under semiquiescent (minimal activities) and then simulated active office conditions to estimate secondary aerosolization of B anthracis spores. </s> Nominal size characteristics, airborne concentrations, and surface contamination of B anthracis particles (colony-forming units) were evaluated. </s> RESULTS Viable B anthracis spores reaerosolized under semiquiescent conditions, with a marked increase in reaerosolization during simulated active office conditions. </s> Increases were observed for B anthracis collected on open sheep blood agar plates (P<.001) and personal air monitors (P =.01) during active office conditions. </s> More than 80% of the B anthracis particles collected on stationary monitors were within an alveolar respirable size range of 0.95 to 3.5 micro m.   CONCLUSIONS Bacillus anthracis spores used in a recent terrorist incident reaerosolized under common office activities. </s> These findings have important implications for appropriate respiratory protection, remediation, and reoccupancy of contaminated office environments.", 'OBJECTIVE To determine rates of suicide associated with pregnancy by the type of pregnancy. </s> DESIGN Register linkage study. </s> Information on suicides in women of reproductive age was linked with the Finnish birth, abortion, and hospital discharge registers to find out how many women who committed suicide had had a completed pregnancy during her last year of life. </s> SETTING Nationwide data from Finland. </s> SUBJECTS Women who committed suicide in 1987-94. </s> RESULTS There were 73 suicides associated with pregnancy, representing 5.4% of all suicides in women in this age group. </s> The mean annual suicide rate was 11.3 per 100,000. </s> The suicide rate associated with birth was significantly lower (5.9) and the rates associated with miscarriage (18.1) and induced abortion (34.7) were significantly higher than in the population. </s> The risk associated with birth was higher among teenagers and that associated with abortion was increased in all age groups. </s> Women who had committed a suicide tended to come from lower social classes and were more likely to be unmarried than other women who had had a completed pregnancy. </s> CONCLUSIONS The increased risk of suicide after an induced abortion indicates either common risk factors for both or harmful effects of induced abortion on mental health.'], 'label': ['00000010', '00000001000'], 'stance': tensor([1, 2])}
'''

def get_evi_role_dict():
    evidence_role = jsonlines.open('../data/para_scifact/scifact_all_evidence_with_role.jsonl', 'r')
    evi_role = {}
    for line in evidence_role:
        evi_role[line["id"]] = line["roles"]
    return evi_role

def pack_arg_feature_tensor(roles):
    max_length = max([len(i) for i in roles])
    #print(roles, max_length)
    role_type = ['OBJECTIVE', 'BACKGROUND', "METHODS", "RESULTS", "CONCLUSIONS", 'none']
    features = np.zeros((len(roles),max_length,len(role_type)), dtype=np.float)
    for i,role in enumerate(roles):
        for j,r in enumerate(role):
            if r not in role_type:
                continue
            features[i, j, role_type.index(r)] = np.float(1.0)
    return features

def get_batch_arg_feature(batch, evi_role):
    # 'doc_id': tensor([33872649, 25641414])
    doc_ids = batch["doc_id"].numpy()
    roles = []
    for doc in doc_ids:
        #print(doc, evi_role[doc])
        roles.append(evi_role[doc])
    return doc_ids, roles


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--repfile', type=str, default = "roberta-large", help="Word embedding file")
    argparser.add_argument('--corpus_file', type=str, default="../data/para_scifact/corpus.jsonl")
    argparser.add_argument('--train_file', type=str, default="../data/para_scifact/claims_train_tfidf_retrieved.jsonl")
    argparser.add_argument('--test_file', type=str, default="../data/para_scifact/claims_dev_tfidf_retrieved.jsonl")

    #argparser.add_argument('--train_file', type=str)
    argparser.add_argument('--pre_trained_model', type=str)
    argparser.add_argument('--bert_lr', type=float, default=1e-5, help="Learning rate for BERT-like LM")
    argparser.add_argument('--lr', type=float, default=5e-6, help="Learning rate")
    argparser.add_argument('--dropout', type=float, default=0, help="embedding_dropout rate")
    argparser.add_argument('--bert_dim', type=int, default=1024, help="bert_dimension")
    argparser.add_argument('--train_batch_size', type=int, default=2) # roberta-large: 2; bert: 8
    argparser.add_argument('--eval_batch_size', type=int, default=2) # roberta-large: 2; bert: 8

    argparser.add_argument('--epoch', type=int, default=20, help="Training epoch")
    argparser.add_argument('--max_seq_length', type=int, default=512)
    argparser.add_argument('--update_step', type=int, default=10)
    argparser.add_argument('--k', type=int, default=3)

    argparser.add_argument('--output_dir', type=str, default = "../model/arg_role_feature_paragraph_encoding_rationale_selection_model")
    # argparser.add_argument('--prediction', type=str, default = "./model/arg_role_feature_paragraph_encoding_rationale_selection_model/prediction_scifact_roberta_rationale_paragraph.jsonl")
    performance_file = 'eval_score.txt'
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    reset_random_seed(12345)

    args = argparser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.repfile)

    if args.train_file:
        train = True
        print('training mode...')
    else:
        train = False
    if args.test_file:
        test = True
        print('testing mode...')
    else:
        test = False

    params = vars(args)
    print('This time parameters:')
    for k,v in params.items():
        print(k,v)

    output_eval_file = os.path.join(args.output_dir, performance_file)
    with open(output_eval_file, "a") as f:
        print('This time parameters:', file=f)
        for k,v in params.items():
            print(k,v, file=f)

    try:
        os.makedirs(args.output_dir)
    except:
        pass

    if train:
        train_set = SciFactParagraphBatchDataset(args.corpus_file, args.train_file, 
                                                 sep_token = tokenizer.sep_token, k = args.k, dummy=False)
    dev_set = SciFactParagraphBatchDataset(args.corpus_file, args.test_file,
                                           sep_token=tokenizer.sep_token, k=args.k, dummy=False)
    evi_role = get_evi_role_dict()
    # for epoch in range(args.epoch):
    #     tq = tqdm(DataLoader(train_set, batch_size=args.batch_size, shuffle=True))
    #     for i, batch in enumerate(tq):
    #         print('查看batch数据')
    #         print(i, batch)
    #         doc_ids, roles = get_batch_arg_feature(batch, evi_role)
    #         arg_features = pack_arg_feature_tensor(roles)
    #
    # exit(0)

    # 注意调用的时候，需要设置k值，不设置k值的话，系统会默认用cited-doc作为待抽取摘要
    # 设置大于0的k值之后，会使用retrived abstract。

    model = RationaleParagraphClassifier(args.repfile, args.bert_dim,
                                      args.dropout)#.to(device)

    if args.pre_trained_model is not None:
        model.load_state_dict(torch.load(args.pre_trained_model))
        model.reinitialize()    ############
        print("使用预训练模型retrain",args.pre_trained_model)
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
            tq = tqdm(DataLoader(train_set, batch_size = args.train_batch_size, shuffle=True))
            for i, batch in enumerate(tq):
                #print(i, batch)
                doc_ids, roles = get_batch_arg_feature(batch, evi_role)
                #arg_feature = pack_arg_feature_tensor(roles)
                #arg_feature = torch.tensor(arg_feature, dtype=torch.float32).to(device)
                #context_arg_feature = torch.tensor(context_arg_feature, dtype=torch.float32).to(device)

                encoded_dict = encode(tokenizer, batch)
                transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id, args.repfile)
                encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
                transformation_indices = [tensor.to(device) for tensor in transformation_indices]
                padded_rationale_label, rationale_label = batch_rationale_label(batch["label"], padding_idx = 2)
                #print(padded_rationale_label)
                #print(padded_rationale_label.shape)
                rationale_out, loss = \
                    model(encoded_dict=encoded_dict, transformation_indices=transformation_indices,
                          rationale_label = padded_rationale_label.to(device))
                loss.backward()

                if i % args.update_step == args.update_step - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    tq.set_description(f'Epoch {epoch}, iter {i}, loss: {round(loss.item(), 4)}')
            scheduler.step()

            # Evaluation
            with open(output_eval_file, "a") as f:
                train_score = evaluation(model, train_set)
                dev_score = evaluation(model, dev_set)
                print(f'Epoch {epoch}, train rationale f1 p r: %.4f, %.4f, %.4f' % train_score[:-1], file=f)
                print(f'Epoch {epoch}, eval rationale f1 p r: %.4f, %.4f, %.4f' % dev_score[:-1], file=f)
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), file=f)
                dev_perf = dev_score[0]
                if dev_perf >= prev_performance:
                    checkpoint = os.path.join(args.output_dir, 'pytorch_model.bin')
                    torch.save(model.state_dict(), checkpoint)
                    best_state_dict = model.state_dict()
                    prev_performance = dev_perf
                    print("Best eval score! New model saved!", file=f)

    # if test:
    #     if train:
    #         del model
    #         model = JointParagraphClassifier(args.repfile, args.bert_dim,
    #                                           args.dropout).to(device)
    #         model.load_state_dict(best_state_dict)
    #         print("Testing on the new model.")
    #     else:
    #         model.load_state_dict(torch.load(args.checkpoint))
    #         print("Loaded saved model.")
    #
    #     # Evaluation
    #     dev_score = evaluation(model, dev_set)
    #     print(f'Test rationale f1 p r: %.4f, %.4f, %.4f' % dev_score[:-1])
    #     print()
    #     params["rationale_f1"] = dev_score[0]
    #     params["rationale_precision"] = dev_score[1]
    #     params["rationale_recall"] = dev_score[2]
    #
    #     with jsonlines.open(args.log_file, mode='a') as writer:
    #         writer.write(params)
    #
    #     rationale_json = rationale2json(dev_set.samples, dev_score[-1])
    #     #print(rationale_json)
    #
    #     with jsonlines.open(args.prediction, mode='a') as writer:
    #         writer.write(rationale_json)