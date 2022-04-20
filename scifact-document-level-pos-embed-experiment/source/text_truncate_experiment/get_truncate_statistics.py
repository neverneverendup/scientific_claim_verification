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
from util import arg2param, flatten, stance2json, rationale2json, merge_json
from my_models import FreezeStanceArgJointParagraphClassifier, ArgJointParagraphClassifier
from dataset import ArgSciFactParagraphBatchDataset, TruncateArgSciFactParagraphBatchDataset

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
        for batch in tqdm(DataLoader(dataset, batch_size = args.eval_batch_size, shuffle=False)):

            doc_ids, roles = get_batch_arg_feature(batch, evi_role)
            arg_feature, context_arg_feature = pack_arg_feature_tensor(roles)
            arg_feature = torch.tensor(arg_feature, dtype=torch.float32).to(device)

            encoded_dict = encode(tokenizer, batch)
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], 
                                                           tokenizer.sep_token_id, args.repfile)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            rationale_out, stance_out, _, _ = model(encoded_dict, transformation_indices, arg_feature)
            stance_preds.extend(stance_out)                
            rationale_predictions.extend(remove_dummy(rationale_out))
            kept_sen_ids = [i.item() for i in batch['sentence_ids']]
            d = { i:val for i,val in kept_sen_ids}
            order_dicts.append(d)

    return rationale_predictions, stance_preds, order_dicts

def evaluation(model, dataset, dummy=True):
    model.eval()
    rationale_predictions = []
    rationale_labels = []
    stance_preds = []
    stance_labels = []

    print('evaluation dataset')
    #print(dataset.samples)

    def remove_dummy(rationale_out):
        return [out[1:] for out in rationale_out]

    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size = args.eval_batch_size, shuffle=False)):
            #print(batch)
            doc_ids, roles = get_batch_arg_feature(batch, evi_role)
            arg_feature, context_arg_feature = pack_arg_feature_tensor(roles)
            arg_feature = torch.tensor(arg_feature, dtype=torch.float32).to(device)

            encoded_dict = encode(tokenizer, batch)
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], 
                                                               tokenizer.sep_token_id, args.repfile)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            stance_label = batch["stance"].to(device)
            padded_rationale_label, rationale_label = batch_rationale_label(batch["label"], padding_idx = 2)
            rationale_out, stance_out, rationale_loss, stance_loss = \
                model(encoded_dict, transformation_indices, arg_feature, stance_label = stance_label,
                      rationale_label = padded_rationale_label.to(device))
            stance_preds.extend(stance_out)
            stance_labels.extend(stance_label.cpu().numpy().tolist())

            rationale_predictions.extend(remove_dummy(rationale_out))
            rationale_labels.extend(remove_dummy(rationale_label))

    stance_f1 = f1_score(stance_labels,stance_preds,average="micro",labels=[1, 2])
    stance_precision = precision_score(stance_labels,stance_preds,average="micro",labels=[1, 2])
    stance_recall = recall_score(stance_labels,stance_preds,average="micro",labels=[1, 2])
    rationale_f1 = f1_score(flatten(rationale_labels),flatten(rationale_predictions))
    rationale_precision = precision_score(flatten(rationale_labels),flatten(rationale_predictions))
    rationale_recall = recall_score(flatten(rationale_labels),flatten(rationale_predictions))
    return stance_f1, stance_precision, stance_recall, rationale_f1, rationale_precision, rationale_recall

samples_num = 0
truncated_num = 0
avg_truncated_rationale_seg_length = 0.0
avg_rationale_length = 0.0
avg_truncated_ratio = 0.0
total_rational_num = 0.0
avg_truncated_rational_length = 0.0

def encode(tokenizer, batch, max_sent_len = 512):
    def truncate(input_ids, max_length, sep_token_id, pad_token_id,labels):
        #print(input_ids)
        #print(tokenizer.decode(input_ids[0]))
        def longest_first_truncation(sentences, objective, labels):
            global truncated_num
            global avg_truncated_rationale_seg_length
            global avg_truncated_ratio
            global avg_rationale_length
            global total_rational_num
            global avg_truncated_rational_length

            sent_lens = [len(sent) for sent in sentences]
            each_sen_original_length_d = {}
            each_sen_truncate_length_d = {}
            each_sen_truncate_ratio_d = {}
            is_truncated = False
            for i in range(len(sent_lens)):
                #print(sentences[i])
                #print(tokenizer.decode(sentences[i]))
                each_sen_truncate_length_d[i] = 0.0
                each_sen_original_length_d[i] = sent_lens[i]
                each_sen_truncate_ratio_d[i] = 0.0

            while np.sum(sent_lens) > objective:
                max_position = np.argmax(sent_lens)
                sent_lens[max_position] -= 1
                # 这一句被裁的token数量加一
                each_sen_truncate_length_d[max_position]+=1

            evidence_sentence_ids = []
            for id,val in enumerate(labels):
                if val=='1':
                    evidence_sentence_ids.append(id)
                    avg_rationale_length += sent_lens[id]
                    total_rational_num += 1

            if len(evidence_sentence_ids) != 0:
                print(labels, evidence_sentence_ids,sent_lens)

                # 被裁rationale数量
                truncate_rationales = 0
                avg_rationale_truncate_length = 0.0
                rationale_truncate_ratio = 0.0

                for i in range(len(sent_lens)):
                    each_sen_truncate_ratio_d[i] = (each_sen_truncate_length_d[i] / each_sen_original_length_d[i])
                    if i in evidence_sentence_ids and each_sen_truncate_ratio_d[i] > 0.01:
                        truncate_rationales += 1
                        avg_rationale_truncate_length += each_sen_truncate_length_d[i]
                        avg_truncated_rationale_seg_length += each_sen_truncate_length_d[i]
                        avg_truncated_ratio += each_sen_truncate_ratio_d[i]
                        #print(each_sen_truncate_ratio_d[i])
                        avg_truncated_rational_length += each_sen_original_length_d[i]
                        #arg_role[evi_role[doc_id][i]] += 1

                if truncate_rationales!=0:
                    is_truncated = True
                    truncated_num += 1
                    # 本篇文档被裁rationale的句子数占总rationale的比例
                    rationale_truncate_ratio = truncate_rationales / len(evidence_sentence_ids)
                    # 本篇文档被裁rationale的平均长度
                    avg_rationale_truncate_length = avg_rationale_truncate_length /truncate_rationales
                    print('truncate_rationales %s, avg_rationale_truncate_length %s, rationale_truncate_ratio %s '%(truncate_rationales, avg_rationale_truncate_length,
                        rationale_truncate_ratio))
            return [sentence[:length] for sentence, length in zip(sentences, sent_lens)], is_truncated

        all_paragraphs = []
        global samples_num
        for id, paragraph in enumerate(input_ids):
            samples_num +=1

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
                truncated_sentences, is_truncated = longest_first_truncation(idx_by_sentence[1:], objective, labels[id])
                truncated_paragraph = torch.cat([idx_by_sentence[0]] + truncated_sentences + [torch.tensor([sep_token_id])],0)
                all_paragraphs.append(truncated_paragraph.unsqueeze(0))

        return torch.cat(all_paragraphs, 0)

    inputs = list(zip(batch["claim"], batch["paragraph"]))
    # batch["label"]
    # for label in labels])
    # print(inputs)

    #print(batch["claim_id"])
    encoded_dict = tokenizer.batch_encode_plus(
        inputs,
        pad_to_max_length=True,add_special_tokens=True,
        return_tensors='pt')
    #print(tokenizer.decode(encoded_dict['input_ids'][0]))
    if encoded_dict['input_ids'].size(1) > max_sent_len:
        if 'token_type_ids' in encoded_dict:
            encoded_dict = {
                "input_ids": truncate(encoded_dict['input_ids'], max_sent_len, 
                                      tokenizer.sep_token_id, tokenizer.pad_token_id, batch['label']),
                'token_type_ids': encoded_dict['token_type_ids'][:,:max_sent_len],
                'attention_mask': encoded_dict['attention_mask'][:,:max_sent_len]
            }
        else:
            encoded_dict = {
                "input_ids": truncate(encoded_dict['input_ids'], max_sent_len, 
                                      tokenizer.sep_token_id, tokenizer.pad_token_id,batch['label']),
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

def get_evi_role_dict():
    evidence_role = jsonlines.open('../../data/para_scifact/scifact_all_evidence_with_role.jsonl', 'r')
    #evidence_role = jsonlines.open('../../data/para_scifact/update_scifact_all_evidence_with_role.jsonl', 'r')

    evi_role = {}
    for line in evidence_role:
        evi_role[line["id"]] = line["roles"]
    return evi_role

def get_batch_arg_feature(batch, evi_role):
    # 'doc_id': tensor([33872649, 25641414])
    doc_ids = batch["doc_id"].numpy()
    roles = []
    for idx in range(batch["doc_id"].numpy().size):
        dataset = batch["dataset"].numpy()[idx]
        doc_id = batch["doc_id"].numpy()[idx]

        #print(batch)
        #print(dataset)
        # if dataset == 1:
        #     roles.append(['none'] + evi_role[doc_id])
        #     #print(roles)
        #     #print('1')
        # else:
        #     #print('2')
        #     kept_sentences_ids = batch["sentence_ids"][idx].split('_')[:-1]
        #     #print(kept_sentences_ids)
        #     #print(evi_role[doc_id])
        #     roles.append(['none'] + [evi_role[doc_id][int(i)] for i in kept_sentences_ids])
        kept_sen_ids = batch['sentence_ids']

        kept_sen_ids = [i.item() for i in kept_sen_ids]
        print('kept_Sen_ids', kept_sen_ids)
        role = batch["role"]
        role = [i[0] for i in role]
        #print(role)
        roles.append(role)


    return doc_ids, roles

role_type = ['OBJECTIVE', 'BACKGROUND', "METHODS", "RESULTS", "CONCLUSIONS", 'none']
def pack_arg_feature_tensor(roles):
    max_length = max([len(i) for i in roles])
    context_features = np.zeros((len(roles),max_length,3*len(role_type)), dtype=np.float)
    features = np.zeros((len(roles),max_length,len(role_type)), dtype=np.float)
    for i,role in enumerate(roles):
        for j,r in enumerate(role):
            if r not in role_type:
                continue
            features[i, j, role_type.index(r)] = np.float(1.0)
    #print(context_features)
    return features, context_features

if __name__ == "__main__":



    argparser = argparse.ArgumentParser(description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--repfile', type=str, default = "roberta-large", help="Word embedding file")
    argparser.add_argument('--train_file', type=str, default="../../data/para_scifact/claims_train_biosent_retrieved.jsonl")
    argparser.add_argument('--test_file', type=str, default="../../data/para_scifact/claims_dev_biosent_retrieved.jsonl")
    argparser.add_argument('--dataset', type=str, default="../../data/para_scifact/claims_dev.jsonl")
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

    argparser.add_argument('--update_step', type=int, default=10)
    argparser.add_argument('--k', type=int, default=12) # 注意一定要设置这个k_train，大一点好，增加鲁棒性
    argparser.add_argument('--downsample_n', type=int, default=0)
    argparser.add_argument('--downsample_p', type=float, default=0.5)
    
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    reset_random_seed(12345)

    args = argparser.parse_args()
    performance_file = 'eval_score.txt'
    output_eval_file = os.path.join(args.output_dir, performance_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.repfile)

    try:
        os.makedirs(args.output_dir)
    except:
        pass

    params = vars(args)
    with open(output_eval_file, 'a', encoding='utf-8')as f:
        for k,v in params.items():
            print(k,v)
            print(k,v,file=f)

    evi_role = get_evi_role_dict()


    train_set = TruncateArgSciFactParagraphBatchDataset(args.corpus_file, args.train_file,
                                                 sep_token = tokenizer.sep_token, k = args.k,dummy=False,
                                                 downsample_n = args.downsample_n,
                                                downsample_p = args.downsample_p)

    dev_set = TruncateArgSciFactParagraphBatchDataset(args.corpus_file, args.test_file,dummy=False,
                                           sep_token = tokenizer.sep_token, k = args.k, downsample_n=0)

    # train_set = ArgSciFactParagraphBatchDataset(args.corpus_file, args.train_file,
    #                                                     sep_token=tokenizer.sep_token, k=args.k, dummy=False,
    #                                                     downsample_n=args.downsample_n,
    #                                                     downsample_p=args.downsample_p)
    #
    # dev_set =  ArgSciFactParagraphBatchDataset(args.corpus_file, args.test_file, dummy=False,
    #                                                   sep_token=tokenizer.sep_token, k=args.k, downsample_n=0)
    #

    prev_performance = 0.0
    for epoch in range(args.epoch):
        sample_p = schedule_sample_p(epoch, args.epoch)
        tq = tqdm(DataLoader(dev_set, batch_size = args.train_batch_size, shuffle=False))
        for i, batch in enumerate(tq):

            doc_ids, roles = get_batch_arg_feature(batch, evi_role)
            arg_feature, context_arg_feature = pack_arg_feature_tensor(roles)
            arg_feature = torch.tensor(arg_feature, dtype=torch.float32).to(device)
            padded_rationale_label, rationale_label = batch_rationale_label(batch["label"], padding_idx = 2)

            encoded_dict = encode(tokenizer, batch)
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id, args.repfile)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            stance_label = batch["stance"].to(device)
            #break
        #break
    # 10319 1021 102 10% k=12
    # 需要裁断的实例总数 2008 有rationale被踩的实例数量 202 0.10059760956175298
    # 360 202 0.5611 k=0
    print('需要裁断的实例总数',samples_num, '有rationale被踩的实例数量',truncated_num, truncated_num/samples_num)
    print('平均每条证据裁断token数%s 平均每条证据裁断比例%s'%(avg_truncated_rationale_seg_length/truncated_num, avg_truncated_ratio/truncated_num))
    print('rationale总数',total_rational_num, '平均rationale长度',avg_rationale_length/total_rational_num, '平均被裁rationale长度',avg_truncated_rational_length/truncated_num)
    # 758 60 8% k=12
    # 748 60 8%
    # 104 60 57.7% k=0

    # rationale_predictions, stance_preds, order_dicts = predict(model, test_set)
    # rationale_json = truncate_rationale2json(order_dicts, test_set.samples, rationale_predictions)




