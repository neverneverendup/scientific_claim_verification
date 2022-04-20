import argparse

import torch
import jsonlines

import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score

import random
import numpy as np

from tqdm import tqdm
from util import flatten, stance2json, rationale2json, merge_json
from paragraph_model_dynamic import ArgJointParagraphClassifier
from dataset import SciFactParagraphBatchDataset

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

    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size = args.batch_size, shuffle=False)):

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

    return rationale_predictions, stance_preds

def evaluation(model, dataset, dummy=True):
    model.eval()
    rationale_predictions = []
    rationale_labels = []
    stance_preds = []
    stance_labels = []
    
    def remove_dummy(rationale_out):
        return [out[1:] for out in rationale_out]
        

    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size = args.batch_size*5, shuffle=False)):

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

def post_process_stance(rationale_json, stance_json):
    assert(len(rationale_json) == len(stance_json))
    for stance_pred, rationale_pred in zip(stance_json, rationale_json):
        assert(stance_pred["claim_id"] == rationale_pred["claim_id"])
        for doc_id, pred in rationale_pred["evidence"].items():
            if len(pred) == 0:
                stance_pred["labels"][doc_id]["label"] = "NOT_ENOUGH_INFO"
    return stance_json

def get_evi_role_dict():
    evidence_role = jsonlines.open('data/para_scifact/scifact_all_evidence_with_role.jsonl', 'r')
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
        if dataset == 1:
            roles.append(['none'] + evi_role[doc_id])
            #print(roles)
            #print('1')
        else:
            #print('2')

            kept_sentences_ids = batch["sentence_ids"][idx].split('_')[:-1]
            #print(kept_sentences_ids)
            #print(evi_role[doc_id])
            roles.append(['none'] + [evi_role[doc_id][int(i)] for i in kept_sentences_ids])
    #print(roles)
    return doc_ids, roles

def pack_arg_feature_tensor(roles):
    max_length = max([len(i) for i in roles])
    #print(roles, max_length)
    role_type = ['OBJECTIVE', 'BACKGROUND', "METHODS", "RESULTS", "CONCLUSIONS", 'none']
    context_features = np.zeros((len(roles),max_length,3*len(role_type)), dtype=np.float)
    features = np.zeros((len(roles),max_length,len(role_type)), dtype=np.float)
    for i,role in enumerate(roles):
        for j,r in enumerate(role):
            if r not in role_type:
                continue
            features[i, j, role_type.index(r)] = np.float(1.0)

    # for i,para in enumerate(features):
    #     for j,sen in enumerate(para):
    #         if j == 0:
    #             last = np.zeros(len(role_type))
    #         else:
    #             last = para[j-1]
    #         if j == len(para)-1:
    #             next = np.zeros(len(role_type))
    #         else:
    #             next = para[j+1]
    #
    #         this = sen
    #         context_features[i,j,:5] = last
    #         context_features[i,j,5:10] = this
    #         context_features[i,j,10:] = next
            #print(last, sen, next)

    #print(context_features)
    return features, context_features

# def pack_arg_context_feature_tensor(roles):
#     max_length = max([len(i) for i in roles])
#     #print(roles, max_length)
#     role_type = ['OBJECTIVE', 'BACKGROUND', "METHODS", "RESULTS", "CONCLUSIONS"]
#     features = np.zeros((len(roles),max_length,3*len(role_type)), dtype=np.float)
#     for i,role in enumerate(roles):
#         for j,r in enumerate(role):
#             if r not in role_type:
#                 continue
#             features[i, j, role_type.index(r)] = np.float(1.0)
#
#     #print(features)
#     return features

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train, cross-validate and run sentence sequence tagger")
    argparser.add_argument('--repfile', type=str, default = "roberta-large", help="Word embedding file")
    argparser.add_argument('--corpus_file', type=str, default="./data/para_scifact/corpus.jsonl")
    argparser.add_argument('--train_file', type=str, default="./data/para_scifact/claims_train_biosent_retrieved.jsonl")
    argparser.add_argument('--pre_trained_model', type=str)
    #argparser.add_argument('--train_file', type=str)
    argparser.add_argument('--test_file', type=str, default="./data/para_scifact/claims_dev_biosent_retrieved.jsonl")
    argparser.add_argument('--dataset', type=str, default="./data/para_scifact/claims_dev.jsonl")
    argparser.add_argument('--bert_lr', type=float, default=1e-5, help="Learning rate for BERT-like LM")
    argparser.add_argument('--lr', type=float, default=5e-6, help="Learning rate")
    argparser.add_argument('--dropout', type=float, default=0, help="embedding_dropout rate")
    argparser.add_argument('--bert_dim', type=int, default=1024, help="bert_dimension")
    argparser.add_argument('--epoch', type=int, default=15, help="Training epoch")
    argparser.add_argument('--MAX_SENT_LEN', type=int, default=512)
    argparser.add_argument('--loss_ratio', type=float, default=6)
    folder = './arg_scifact_model/arg_feature_'
    argparser.add_argument('--checkpoint', type=str, default = folder+"scifact_roberta_joint_paragraph_dynamic.model")
    argparser.add_argument('--log_file', type=str, default = folder+"joint_paragraph_roberta_dynamic_performances.jsonl")
    argparser.add_argument('--prediction', type=str, default = folder+"prediction_scifact_roberta_joint_paragraph_dynamic.jsonl")
    argparser.add_argument('--update_step', type=int, default=10)
    argparser.add_argument('--batch_size', type=int, default=1) # roberta-large_16bs_1e-5lr: 2; bert: 8
    argparser.add_argument('--k', type=int, default=12) # 注意一定要设置这个k_train，大一点好，增加鲁棒性
    argparser.add_argument('--downsample_n', type=int, default=1)
    argparser.add_argument('--downsample_p', type=float, default=0.5)
    
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    
    reset_random_seed(12345)

    args = argparser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.repfile)

    if args.train_file:
        train = True
        #assert args.repfile is not None, "Word embedding file required for training."
    else:
        train = False

    if args.test_file:
        test = True
    else:
        test = False

    print('train',train,'test', test)

    params = vars(args)

    for k,v in params.items():
        print(k,v)

    evi_role = get_evi_role_dict()

    if train:
        train_set = SciFactParagraphBatchDataset(args.corpus_file, args.train_file, 
                                                 sep_token = tokenizer.sep_token, k = args.k, 
                                                 downsample_n = args.downsample_n, 
                                                downsample_p = args.downsample_p)
    dev_set = SciFactParagraphBatchDataset(args.corpus_file, args.test_file, 
                                           sep_token = tokenizer.sep_token, k = args.k, downsample_n=0)

    model = ArgJointParagraphClassifier(args.repfile, args.bert_dim,
                                      args.dropout)#.to(device)

    if args.pre_trained_model is not None:
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

        prev_performance = 0.0
        for epoch in range(args.epoch):
            sample_p = schedule_sample_p(epoch, args.epoch)
            tq = tqdm(DataLoader(train_set, batch_size = args.batch_size, shuffle=True))
            for i, batch in enumerate(tq):
                #print(batch)

                doc_ids, roles = get_batch_arg_feature(batch, evi_role)
                arg_feature, context_arg_feature = pack_arg_feature_tensor(roles)
                arg_feature = torch.tensor(arg_feature, dtype=torch.float32).to(device)
                encoded_dict = encode(tokenizer, batch)
                transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id, args.repfile)
                encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
                transformation_indices = [tensor.to(device) for tensor in transformation_indices]
                stance_label = batch["stance"].to(device)
                padded_rationale_label, rationale_label = batch_rationale_label(batch["label"], padding_idx = 2)
                rationale_out, stance_out, rationale_loss, stance_loss = \
                    model(encoded_dict, transformation_indices, arg_feature, stance_label = stance_label,
                          rationale_label = padded_rationale_label.to(device), sample_p = sample_p)
                rationale_loss *= args.loss_ratio
                loss = rationale_loss + stance_loss
                loss.backward()

                if i % args.update_step == args.update_step - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    tq.set_description(f'Epoch {epoch}, iter {i}, loss: {round(loss.item(), 4)}, stance loss: {round(stance_loss.item(), 4)}, rationale loss: {round(rationale_loss.item(), 4)}')
            scheduler.step()

            # Evaluation
            train_score = evaluation(model, train_set)
            print(f'Epoch {epoch}, train stance f1 p r: %.4f, %.4f, %.4f, rationale f1 p r: %.4f, %.4f, %.4f' % train_score)

            dev_score = evaluation(model, dev_set)
            print(f'Epoch {epoch}, dev stance f1 p r: %.4f, %.4f, %.4f, rationale f1 p r: %.4f, %.4f, %.4f' % dev_score)
            torch.save(model.state_dict(), args.checkpoint) ############

            dev_perf = dev_score[0] * dev_score[3]
            model_path = folder + str(epoch) + '_stance_f1_' + str(
                dev_score[0]) + '_rationale_f1_' + str(dev_score[3]) + '.model'

            if dev_perf >= prev_performance:
               torch.save(model.state_dict(), model_path)
               print(model_path)
               best_state_dict = model.state_dict()
               prev_performance = dev_perf
               print("New model saved!")
            else:
               print("Skip saving model.")

    if test:
        if train:
            del model
        model = ArgJointParagraphClassifier(args.repfile, args.bert_dim,
                                          args.dropout).to(device)
        model.load_state_dict(torch.load(args.checkpoint))
        
        # Evaluation
        #dev_score = evaluation(model, dev_set)
        #print(f'Test stance f1 p r: %.4f, %.4f, %.4f, rationale f1 p r: %.4f, %.4f, %.4f' % dev_score)
        
        dev_set = SciFactParagraphBatchDataset(args.corpus_file, args.test_file, 
                                       sep_token = tokenizer.sep_token, k = args.k, downsample_n=0, train=False)
        
        rationale_predictions, stance_preds = predict(model, dev_set)
        rationale_json = rationale2json(dev_set.samples, rationale_predictions)
        stance_json = stance2json(dev_set.samples, stance_preds)
        stance_json = post_process_stance(rationale_json, stance_json)
        merged_json = merge_json(rationale_json, stance_json)
        
        with jsonlines.open(args.prediction, 'w') as output:
            for result in merged_json:
                output.write(result)

        data = GoldDataset(args.corpus_file, args.dataset)
        predictions = PredictedDataset(data, args.prediction)
        res = metrics.compute_metrics(predictions)
        params["evaluation"] = res
        with jsonlines.open(args.log_file, mode='a') as writer:
            writer.write(params)
