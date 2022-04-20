import argparse
import torch
print('torch 版本号:',torch.__version__)
import jsonlines
import os
import time
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score
import random
import numpy as np
from tqdm import tqdm
from util import arg2param, flatten, stance2json, rationale2json, merge_json,truncate_rationale2json
from my_models import ArgJointParagraphClassifier, JointParagraphClassifier
from dataset import RefinedAbstract_SciFactParagraphBatchDataset, TruncateArgSciFactParagraphBatchDataset, my_collcate_fn

import logging

from lib.data import GoldDataset, PredictedDataset
from lib import metrics

def schedule_sample_p(epoch, total):
    return np.sin(0.5* np.pi* epoch / (total-1))

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
    stance_preds = []

    def remove_dummy(rationale_out):
        return [out[1:] for out in rationale_out]

    order_dicts = []
    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,collate_fn=my_collcate_fn)):


            encoded_dict = encode(tokenizer, batch)
            #print('input_ids.shape', encoded_dict["input_ids"].shape)
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"],
                                                           tokenizer.sep_token_id, args.repfile)
            #print('batch_indices', transformation_indices[0].shape)
            #print(batch)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]

            rationale_out, stance_out, _, _ = model(encoded_dict, transformation_indices)
            stance_preds.extend(stance_out)
            rationale_predictions.extend(remove_dummy(rationale_out))

            for sen_ids in batch['sentence_ids']:
                #print(sen_ids)
                d = {i: val for i, val in enumerate(sen_ids)}
                order_dicts.append(d)
                #print(d)
    print(len(rationale_predictions), len(stance_preds),len(order_dicts))
    return rationale_predictions, stance_preds, order_dicts

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

    inputs = list(zip(batch["claim"], batch["paragraph"]))
    encoded_dict = tokenizer.batch_encode_plus(
        inputs,
        pad_to_max_length=True,add_special_tokens=True,
        return_tensors='pt')
    if encoded_dict['input_ids'].size(1) > max_sent_len:
        #print('truncate!!!')
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
    #print('sep tokens',sep_tokens,sep_tokens.shape)
    paragraph_lens = torch.sum(sep_tokens,1).numpy().tolist()
    #print(paragraph_lens)
    indices = torch.arange(sep_tokens.size(-1)).unsqueeze(0).expand(sep_tokens.size(0),-1)
    sep_indices = torch.split(indices[sep_tokens],paragraph_lens)
    #print('sep indices',sep_indices)

    paragraph_lens = []
    all_word_indices = []
    for paragraph in sep_indices:
        #print(paragraph)
        if "large" in model_name:
            paragraph = paragraph[1:]
        #print(paragraph)
        word_indices = [torch.arange(paragraph[i]+1, paragraph[i+1]+1) for i in range(paragraph.size(0)-1)]
        #print(word_indices)
        paragraph_lens.append(len(word_indices))
        all_word_indices.extend(word_indices)
    indices_by_sentence = nn.utils.rnn.pad_sequence(all_word_indices, batch_first=True, padding_value=padding_idx)
    indices_by_sentence_split = torch.split(indices_by_sentence,paragraph_lens)
    indices_by_batch = nn.utils.rnn.pad_sequence(indices_by_sentence_split, batch_first=True, padding_value=padding_idx)
    batch_indices = torch.arange(sep_tokens.size(0)).unsqueeze(-1).unsqueeze(-1).expand(-1,indices_by_batch.size(1),indices_by_batch.size(-1))
    mask = (indices_by_batch>=0)

    #exit(0)
    return batch_indices.long(), indices_by_batch.long(), mask.long()

def post_process_stance(rationale_json, stance_json):
    assert(len(rationale_json) == len(stance_json))
    for stance_pred, rationale_pred in zip(stance_json, rationale_json):
        assert(stance_pred["claim_id"] == rationale_pred["claim_id"])
        for doc_id, pred in rationale_pred["evidence"].items():
            if len(pred) == 0:
                stance_pred["labels"][doc_id]["label"] = "NOT_ENOUGH_INFO"
    return stance_json

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--repfile', type=str, default = "roberta-large", help="Word embedding file")
    argparser.add_argument('--train_file', type=str, default="../../data/para_scifact/claims_train_biosent_retrieved.jsonl")
    argparser.add_argument('--test_file', type=str, default="../../data/para_scifact/claims_dev_biosent_retrieved.jsonl")
    argparser.add_argument('--dataset', type=str, default="../../data/para_scifact/claims_dev.jsonl")
    argparser.add_argument('--scores_dataset', type=str, default="k30_scores_baseline_predicted_rationale_selection.jsonl")

    argparser.add_argument('--corpus_file', type=str, default="../../data/para_scifact/corpus.jsonl")
    argparser.add_argument('--pre_trained_model', type=str)

    argparser.add_argument('--bert_lr', type=float, default=1e-5, help="Learning rate for BERT-like LM")
    argparser.add_argument('--lr', type=float, default=5e-6, help="Learning rate")
    argparser.add_argument('--dropout', type=float, default=0, help="embedding_dropout rate")
    argparser.add_argument('--bert_dim', type=int, default=1024, help="bert_dimension")
    argparser.add_argument('--epoch', type=int, default=15, help="Training epoch")
    argparser.add_argument('--max_seq_length', type=int, default=512)
    argparser.add_argument('--loss_ratio', type=float, default=6)

    argparser.add_argument('--sent_repr_dim', type=int, default=1024, help="sentence_representation_dimension")
    argparser.add_argument('--train_batch_size', type=int, default=2) # roberta-large: 2; bert: 8
    argparser.add_argument('--eval_batch_size', type=int, default=10) # roberta-large: 2; bert: 8
    argparser.add_argument('--output_dir', type=str, default = "./model/debug_joint_arg_model_k12")
    argparser.add_argument('--checkpoint', type=str)

    argparser.add_argument('--update_step', type=int, default=10)
    argparser.add_argument('--k', type=int, default=12) # 注意一定要设置这个k_train，大一点好，增加鲁棒性
    argparser.add_argument('--downsample_n', type=int, default=1)
    argparser.add_argument('--downsample_p', type=float, default=0.5)
    
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    reset_random_seed(12345)

    args = argparser.parse_args()
    performance_file = 'eval_score.txt'
    output_eval_file = os.path.join(args.output_dir, performance_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.repfile)

    if args.train_file:
        train = True
    else:
        train = False

    if args.test_file:
        test = True
    else:
        test = False

    print('train',train,'test', test)

    try:
        os.makedirs(args.output_dir)
    except:
        pass

    params = vars(args)
    with open(output_eval_file, 'a', encoding='utf-8')as f:
        for k,v in params.items():
            print(k,v)
            print(k,v,file=f)

    model = JointParagraphClassifier(bert_path=args.repfile, bert_dim=args.bert_dim, dropout = args.dropout)#.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model = model.to(device)

    if test:
        test_set = RefinedAbstract_SciFactParagraphBatchDataset(corpus=args.corpus_file, claims=args.test_file, scores_dataset=args.scores_dataset,
                                                          sep_token=tokenizer.sep_token, k = 30,
                                                          downsample_n=0,train=False)

        rationale_predictions, stance_preds, order_dicts = predict(model, test_set)
        rationale_json = truncate_rationale2json(order_dicts, test_set.samples, rationale_predictions)

        stance_json = stance2json(test_set.samples, stance_preds)
        stance_json = post_process_stance(rationale_json, stance_json)

        with jsonlines.open(args.output_dir + '/rationale_predictions.jsonl', 'w') as output:
            for result in rationale_json:
                output.write(result)

        with jsonlines.open(args.output_dir + '/label_predictions.jsonl', 'w') as output:
            for result in stance_json:
                output.write(result)

        merged_json = merge_json(rationale_json, stance_json)
        prediction_file = args.output_dir + '/merged_predictions.jsonl'

        with jsonlines.open(prediction_file, 'w') as output:
            for result in merged_json:
                output.write(result)

        data = GoldDataset(args.corpus_file, args.dataset)
        predictions = PredictedDataset(data, prediction_file)
        res = metrics.compute_metrics(predictions)
        params["evaluation"] = res
        with open(output_eval_file, 'a', encoding='utf-8') as f:
            print(params)
            print(params, file=f)



