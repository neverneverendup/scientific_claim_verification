import os
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from pathlib import Path
import numpy as np
import jsonlines
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from utils import token_idx_by_sentence, get_rationale_label, flatten, remove_dummy

from embedding.jointmodel import JointModelClassifier, AbstractRationaleJointModelClassifier, ArgJointModelClassifier
#from embedding.model import JointModelClassifier
# from embedding.AutomaticWeightedLoss import AutomaticWeightedLoss
from evaluation.evaluation_model import evaluation_joint,abstract_rationale_evaluation_joint, evaluation_abstract_retrieval
from dataset.encode import encode_paragraph
from utils import token_idx_by_sentence, get_rationale_label
from get_prediction import get_predictions
from dataset.utils import merge, merge_retrieval
from utils import predictions2jsonl, retrieval2jsonl
from evaluation.evaluation_model import merge_rationale_label, evaluate_rationale_selection, evaluate_label_predictions



def schedule_sample_p(epoch, total):
    if epoch == total-1:
        abstract_sample = 1.0
    else:
        abstract_sample = np.tanh(0.5 * np.pi * epoch / (total-1-epoch))
    rationale_sample = np.sin(0.5 * np.pi * epoch / (total-1))
    return abstract_sample, rationale_sample


def train_base(train_set, dev_set, args):
    # awl = AutomaticWeightedLoss(3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tmp_dir = os.path.join(os.path.curdir, 'tmp-runs/')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = JointModelClassifier(args)
    if args.pre_trained_model is not None:
        model.load_state_dict(torch.load(args.pre_trained_model))
        model.reinitialize()
    model = model.to(device)
    parameters = [{'params': model.bert.parameters(), 'lr': args.bert_lr},
                  {'params': model.abstract_retrieval.parameters(), 'lr': 5e-6}]
    for module in model.extra_modules:
        parameters.append({'params': module.parameters(), 'lr': args.lr})
    optimizer = torch.optim.Adam(parameters)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epochs)
    performance_file = 'eval_score.txt'
    output_eval_file = os.path.join(args.output_dir, performance_file)

    best_f1 = 0
    best_model = model
    model.train()
    #checkpoint = os.path.join(args.save, f'JointModel.model')
    checkpoint = os.path.join(args.output_dir, f'JointModel.model')
    for epoch in range(args.epochs):
        abstract_sample, rationale_sample = schedule_sample_p(epoch, args.epochs)
        model.train()  # cudnn RNN backward can only be called in training mode
        t = tqdm(DataLoader(train_set, batch_size=1, shuffle=True))
        for i, batch in enumerate(t):
            encoded_dict = encode_paragraph(tokenizer, batch['claim'], batch['abstract'])
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id,
                                                           args.model)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            padded_label, rationale_label = get_rationale_label(batch["sentence_label"], padding_idx=2)
            _, _, abstract_loss, rationale_loss, sim_loss, bce_loss = model(encoded_dict, transformation_indices,
                                                                  abstract_label=batch['abstract_label'].to(device),
                                                                  rationale_label=padded_label.to(device),
                                                                  retrieval_label=batch['sim_label'].to(device),
                                                                  train=True, rationale_sample=rationale_sample)
            rationale_loss *= args.lambdas[2]
            abstract_loss *= args.lambdas[1]
            sim_loss *= args.lambdas[0]
            bce_loss = args.alpha * bce_loss
            loss = abstract_loss + rationale_loss + sim_loss + bce_loss
            # 反向传播用来计算梯度，每次调用同一参数的梯度被累加
            loss.backward()
            if (i + 1) % (args.batch_size_accumulated // args.batch_size_gpu) == 0:
                # opt.step()用来利用优化器保存的梯度更新参数
                optimizer.step()
                optimizer.zero_grad()
                t.set_description(f'Epoch {epoch}, iter {i}, loss: {round(loss.item(), 4)},'
                                  f' abstract loss: {round(abstract_loss.item(), 4)},'
                                  f' rationale loss: {round(rationale_loss.item(), 4)},'
                                  f' retrieval loss: {round(sim_loss.item(), 4)},'
                                  f' BCE loss: {round(bce_loss.item(), 4)}')
        scheduler.step()
        train_score = evaluation_joint(model, train_set, args, tokenizer)
        print(f'Epoch {epoch} train abstract score:', train_score[0],
              f'Epoch {epoch} train rationale score:', train_score[1])
        dev_score = evaluation_joint(model, dev_set, args, tokenizer)
        print(f'Epoch {epoch} dev abstract score:', dev_score[0],
              f'Epoch {epoch} dev rationale score:', dev_score[1])

        with open(output_eval_file, 'a', encoding='utf-8') as f:
            print(f'Epoch {epoch} train abstract score:', train_score[0],
                  f'Epoch {epoch} train rationale score:', train_score[1], file=f)
            print(f'Epoch {epoch} dev abstract score:', dev_score[0],
                  f'Epoch {epoch} dev rationale score:', dev_score[1], file=f)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), file=f)

        # save
        # save_path = os.path.join(tmp_dir, str(int(time.time() * 1e5))
        #                          + f'-abstract_f1-{int(dev_score[0]["f1"]*1e4)}'
        #                          + f'-rationale_f1-{int(dev_score[1]["f1"]*1e4)}.model')
        # torch.save(model.state_dict(), save_path)
            if (dev_score[0]['f1'] + dev_score[1]['f1']) / 2 >= best_f1:
                best_f1 = (dev_score[0]['f1'] + dev_score[1]['f1']) / 2
                best_model = model
                print("best performance !")
                print("best performance !", file=f)
    torch.save(best_model.state_dict(), checkpoint)
    return checkpoint

def abstract_rationale_train_base(train_set, dev_set, args):
    # awl = AutomaticWeightedLoss(3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tmp_dir = os.path.join(os.path.curdir, 'tmp-runs/')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AbstractRationaleJointModelClassifier(args)
    if args.pre_trained_model is not None:
        model.load_state_dict(torch.load(args.pre_trained_model))
        model.reinitialize()
    model = model.to(device)
    parameters = [{'params': model.bert.parameters(), 'lr': args.bert_lr},
                  {'params': model.abstract_retrieval.parameters(), 'lr': 5e-6}]
    for module in model.extra_modules:
        parameters.append({'params': module.parameters(), 'lr': args.lr})
    optimizer = torch.optim.Adam(parameters)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epochs)
    performance_file = 'eval_score.txt'
    output_eval_file = os.path.join(args.output_dir, performance_file)
    best_f1 = 0
    best_model = model
    model.train()
    #checkpoint = os.path.join(args.save, f'JointModel.model')
    checkpoint = os.path.join(args.output_dir, f'JointModel.model')
    for epoch in range(args.epochs):
        abstract_sample, rationale_sample = schedule_sample_p(epoch, args.epochs)
        model.train()  # cudnn RNN backward can only be called in training mode
        t = tqdm(DataLoader(train_set, batch_size=1, shuffle=True))
        for i, batch in enumerate(t):
            encoded_dict = encode_paragraph(tokenizer, batch['claim'], batch['abstract'])
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id,
                                                           args.model)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            padded_label, rationale_label = get_rationale_label(batch["sentence_label"], padding_idx=2)
            # _, _, abstract_loss, rationale_loss, sim_loss, bce_loss = model(encoded_dict, transformation_indices,
            #                                                       abstract_label=batch['abstract_label'].to(device),
            #                                                       rationale_label=padded_label.to(device),
            #                                                       retrieval_label=batch['sim_label'].to(device),
            #                                                       train=True, rationale_sample=rationale_sample)

            _, rationale_loss, sim_loss, bce_loss = model(encoded_dict, transformation_indices,
                                                                  abstract_label=batch['abstract_label'].to(device),
                                                                  rationale_label=padded_label.to(device),
                                                                  retrieval_label=batch['sim_label'].to(device),
                                                                  train=True, rationale_sample=rationale_sample)
            rationale_loss *= args.lambdas[2]
            #abstract_loss *= args.lambdas[1]
            sim_loss *= args.lambdas[0]
            bce_loss = args.alpha * bce_loss
            loss =  rationale_loss + sim_loss + bce_loss
            # 反向传播用来计算梯度，每次调用同一参数的梯度被累加
            loss.backward()
            if (i + 1) % (args.batch_size_accumulated // args.batch_size_gpu) == 0:
                # opt.step()用来利用优化器保存的梯度更新参数
                optimizer.step()
                optimizer.zero_grad()
                t.set_description(f'Epoch {epoch}, iter {i}, loss: {round(loss.item(), 4)},'
                                  #f' abstract loss: {round(abstract_loss.item(), 4)},'
                                  f' rationale loss: {round(rationale_loss.item(), 4)},'
                                  f' retrieval loss: {round(sim_loss.item(), 4)},'
                                  f' BCE loss: {round(bce_loss.item(), 4)}')
        scheduler.step()
        train_score = abstract_rationale_evaluation_joint(model, train_set, args, tokenizer)
        print(f'Epoch {epoch} train rationale score:', train_score)
        dev_score = abstract_rationale_evaluation_joint(model, dev_set, args, tokenizer)
        print(f'Epoch {epoch} dev rationale score:', dev_score)

        with open(output_eval_file, 'a', encoding='utf-8') as f:
            print(f'Epoch {epoch} train rationale score:', train_score, file=f)
            print(f'Epoch {epoch} dev rationale score:', dev_score, file=f)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), file=f)

            if dev_score['f1'] >= best_f1:
                best_f1 = dev_score['f1']
                best_model = model
                print("best performance !")
                print("best performance !", file=f)
    torch.save(best_model.state_dict(), checkpoint)
    return checkpoint


def get_evi_role_dict():
    evidence_role = jsonlines.open('../data/scifact_all_evidence_with_role.jsonl', 'r')
    #evidence_role = jsonlines.open('../../data/para_scifact/update_scifact_all_evidence_with_role.jsonl', 'r')

    evi_role = {}
    for line in evidence_role:
        evi_role[line["id"]] = line["roles"]
    return evi_role

def get_batch_arg_feature(batch, evi_role):
    # 'doc_id': tensor([33872649, 25641414])
    doc_ids = batch["doc_id"].numpy()
    roles = []
    #print(batch)
    for idx in range(batch["doc_id"].numpy().size):
        #sim_label = batch["sim_label"].numpy()[idx]
        doc_id = batch["doc_id"].numpy()[idx]
        #print(doc_id)
        #if sim_label == 1:
            #print(batch["kept_sentence_ids"][idx])
        kept_sentences_ids = batch["kept_sentence_ids"][idx].split('_')[:-1]
        roles.append([evi_role[doc_id][int(i)] for i in kept_sentences_ids])
            #print([evi_role[doc_id][int(i)] for i in kept_sentences_ids])
        # else:
        #     roles.append(evi_role[doc_id])
        #print(roles)


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

evi_role = get_evi_role_dict()

def arg_evaluation_joint(model, dataset, args, tokenizer, mode='rationale&label'):
    model.eval()
    abstract_targets = []
    rationale_targets = []
    abstract_outputs = []
    rationale_output = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        t = tqdm(DataLoader(dataset, batch_size=args.batch_size_gpu, shuffle=False))
        for i, batch in enumerate(t):
            doc_ids, roles = get_batch_arg_feature(batch, evi_role)
            arg_feature, context_arg_feature = pack_arg_feature_tensor(roles)
            arg_feature = torch.tensor(arg_feature, dtype=torch.float32).to(device)

            encoded_dict = encode_paragraph(tokenizer, batch['claim'], batch['abstract'])
            # encoded = encode_paragraph(tokenizer, batch['claim'], batch['paragraph'])
            # encoded = {key: tensor.to(device) for key, tensor in encoded.items()}
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id,
                                                           args.model)
            # match_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id,
            #                                       args.model, match=True)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            # match_indices = [tensor.to(device) for tensor in match_indices]
            padded_label, rationale_label = get_rationale_label(batch["sentence_label"], padding_idx=2)
            abstract_out, rationale_out, retrieval_out = model(encoded_dict, transformation_indices, arg_feature)
            # loss = abstract_loss + rationale_loss + retrieval_loss
            # if (i + 1) % (args.batch_size_accumulated // args.batch_size_gpu) == 0:
            #     t.set_description(f'iter {i}, loss: {round(loss.item(), 4)},'
            #                       f' abstract loss: {round(abstract_loss.item(), 4)},'
            #                       f' rationale loss: {round(rationale_loss.item(), 4)},'
            #                       f' retrieval loss: {round(retrieval_loss.item(), 4)}')
            # abstract_out = torch.argmax(abstract_score.cpu(), dim=-1).detach().numpy().tolist()
            # rationale_out = torch.argmax(rationale_score.cpu(), dim=-1).detach().numpy().tolist()

            abstract_targets.extend(batch['abstract_label'])
            abstract_outputs.extend(abstract_out)

            rationale_targets.extend(rationale_label)
            rationale_output.extend(rationale_out)
    if mode == 'label':
        return {
            'f1': f1_score(abstract_targets, abstract_outputs, zero_division=0, average='micro', labels=[1, 2]),
            'p': precision_score(abstract_targets, abstract_outputs, zero_division=0, average='micro', labels=[1, 2]),
            'r': recall_score(abstract_targets, abstract_outputs, zero_division=0, average='micro', labels=[1, 2]),
        }
    elif mode == 'rationale':
        return {
            'f1': f1_score(flatten(rationale_targets), flatten(rationale_output), zero_division=0),
            'p': precision_score(flatten(rationale_targets), flatten(rationale_output), zero_division=0),
            'r': recall_score(flatten(rationale_targets), flatten(rationale_output), zero_division=0)
        }
    else:
        return {
            'f1': f1_score(abstract_targets, abstract_outputs, zero_division=0, average='micro', labels=[1, 2]),
            # 'abstract_f1': tuple(f1_score(abstract_targets, abstract_outputs, zero_division=0, average=None)),
            'p': precision_score(abstract_targets, abstract_outputs, zero_division=0, average='micro', labels=[1, 2]),
            'r': recall_score(abstract_targets, abstract_outputs, zero_division=0, average='micro', labels=[1, 2]),
        }, {
            'f1': f1_score(flatten(rationale_targets), flatten(rationale_output), zero_division=0),
            'p': precision_score(flatten(rationale_targets), flatten(rationale_output), zero_division=0),
            'r': recall_score(flatten(rationale_targets), flatten(rationale_output), zero_division=0)
        }

def get_arg_predictions(args, input_set, checkpoint):
    device = torch.device('cuda' if torch.  cuda.is_available() else 'cpu')
    # args.batch_size_gpu = 8
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = ArgJointModelClassifier(args).to(device)
    # model = JointParagraphClassifier(args.model, args.hidden_dim, args.dropout).to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    # for m in model.state_dict().keys():
    #     print(m)
    # p
    abstract_result = []
    rationale_result = []
    retrieval_result = []
    with torch.no_grad():
        for batch in tqdm(DataLoader(input_set, batch_size=10, shuffle=False)):
            doc_ids, roles = get_batch_arg_feature(batch, evi_role)
            arg_feature, context_arg_feature = pack_arg_feature_tensor(roles)
            arg_feature = torch.tensor(arg_feature, dtype=torch.float32).to(device)

            encoded_dict = encode_paragraph(tokenizer, batch['claim'], batch['abstract'])
            # encoded = encode_paragraph(tokenizer, batch['claim'], batch['paragraph'])
            # encoded = {key: tensor.to(device) for key, tensor in encoded.items()}
            transformation_indices = token_idx_by_sentence(encoded_dict['input_ids'], tokenizer.sep_token_id,
                                                           args.model)
            # match_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id,
            #                                       args.model, match=True)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            # match_indices = [tensor.to(device) for tensor in match_indices]
            # abstract_out, rationale_out = model(encoded_dict, transformation_indices, match_indices)
            abstract_out, rationale_out, retrieval_out = model(encoded_dict, transformation_indices, arg_feature)
            abstract_result.extend(abstract_out)
            rationale_result.extend(rationale_out)
            retrieval_result.extend(retrieval_out)

    return abstract_result, rationale_result, retrieval_result

def arg_train_base(train_set, dev_set, test_set, args):
    # awl = AutomaticWeightedLoss(3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tmp_dir = os.path.join(os.path.curdir, 'tmp-runs/')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = ArgJointModelClassifier(args)
    if args.pre_trained_model is not None:
        #model.load_state_dict(torch.load(args.pre_trained_model))
        #model.reinitialize()
        pretrained_dict = torch.load(args.pre_trained_model)
        model_dict = model.state_dict()
        pretrained_dict = { k:v for k,v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.reinitialize()

    model = model.to(device)
    parameters = [{'params': model.bert.parameters(), 'lr': args.bert_lr},
                  {'params': model.abstract_retrieval.parameters(), 'lr': 5e-6}]
    for module in model.extra_modules:
        parameters.append({'params': module.parameters(), 'lr': args.lr})
    optimizer = torch.optim.Adam(parameters)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.epochs)
    performance_file = 'eval_score.txt'
    output_eval_file = os.path.join(args.output_dir, performance_file)

    best_f1 = 0.0
    best_model = model
    model.train()
    #checkpoint = os.path.join(args.save, f'JointModel.model')
    checkpoint = os.path.join(args.output_dir, f'JointModel.model')
    for epoch in range(args.epochs):
        abstract_sample, rationale_sample = schedule_sample_p(epoch, args.epochs)
        model.train()  # cudnn RNN backward can only be called in training mode
        t = tqdm(DataLoader(train_set, batch_size=1, shuffle=True))
        for i, batch in enumerate(t):

            doc_ids, roles = get_batch_arg_feature(batch, evi_role)
            arg_feature, context_arg_feature = pack_arg_feature_tensor(roles)
            arg_feature = torch.tensor(arg_feature, dtype=torch.float32).to(device)

            encoded_dict = encode_paragraph(tokenizer, batch['claim'], batch['abstract'])
            transformation_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id,
                                                           args.model)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            padded_label, rationale_label = get_rationale_label(batch["sentence_label"], padding_idx=2)
            _, _, abstract_loss, rationale_loss, sim_loss, bce_loss = model(encoded_dict, transformation_indices,
                                                                    arg_feature,
                                                                  abstract_label=batch['abstract_label'].to(device),
                                                                  rationale_label=padded_label.to(device),
                                                                  retrieval_label=batch['sim_label'].to(device),
                                                                  train=True, rationale_sample=rationale_sample)
            rationale_loss *= args.lambdas[2]
            abstract_loss *= args.lambdas[1]
            sim_loss *= args.lambdas[0]
            bce_loss = args.alpha * bce_loss
            loss = abstract_loss + rationale_loss + sim_loss + bce_loss
            # 反向传播用来计算梯度，每次调用同一参数的梯度被累加
            loss.backward()
            if (i + 1) % (args.batch_size_accumulated // args.batch_size_gpu) == 0:
                # opt.step()用来利用优化器保存的梯度更新参数
                optimizer.step()
                optimizer.zero_grad()
                t.set_description(f'Epoch {epoch}, iter {i}, loss: {round(loss.item(), 4)},'
                                  f' abstract loss: {round(abstract_loss.item(), 4)},'
                                  f' rationale loss: {round(rationale_loss.item(), 4)},'
                                  f' retrieval loss: {round(sim_loss.item(), 4)},'
                                  f' BCE loss: {round(bce_loss.item(), 4)}')
        scheduler.step()
        train_score = arg_evaluation_joint(model, train_set, args, tokenizer)
        print(f'Epoch {epoch} train abstract score:', train_score[0],
              f'Epoch {epoch} train rationale score:', train_score[1])
        dev_score = arg_evaluation_joint(model, dev_set, args, tokenizer)
        print(f'Epoch {epoch} dev abstract score:', dev_score[0],
              f'Epoch {epoch} dev rationale score:', dev_score[1])

        with open(output_eval_file, 'a', encoding='utf-8') as f:
            print(f'Epoch {epoch} train abstract score:', train_score[0],
                  f'Epoch {epoch} train rationale score:', train_score[1], file=f)
            print(f'Epoch {epoch} dev abstract score:', dev_score[0],
                  f'Epoch {epoch} dev rationale score:', dev_score[1], file=f)

            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), file=f)

            if (dev_score[0]['f1'] + dev_score[1]['f1']) / 2 >= best_f1:
                best_f1 = (dev_score[0]['f1'] + dev_score[1]['f1']) / 2
                best_model = model
                print("best performance !")
                print("best performance !", file=f)
                torch.save(best_model.state_dict(), checkpoint)

        abstract_result, rationale_result, retrieval_result = get_arg_predictions(args, test_set, checkpoint)
        rationales, labels = predictions2jsonl(test_set.samples, abstract_result, rationale_result)
        merge(rationales, labels, args.merge_results)
        print('rationale selection...')
        evaluate_rationale_selection(args, "prediction/rationale_selection.jsonl")
        print('label predictions...')
        evaluate_label_predictions(args, "prediction/label_predictions.jsonl")
        print('merging predictions...')
        res = merge_rationale_label(rationales, labels, args, state='valid', gold=args.gold)

        with open(output_eval_file, 'a', encoding='utf-8') as f:
            import json
            json.dump(res.to_dict(), f, indent=2)


        # save
        # save_path = os.path.join(tmp_dir, str(int(time.time() * 1e5))
        #                          + f'-abstract_f1-{int(dev_score[0]["f1"]*1e4)}'
        #                          + f'-rationale_f1-{int(dev_score[1]["f1"]*1e4)}.model')
        # torch.save(model.state_dict(), save_path)

    torch.save(best_model.state_dict(), checkpoint)
    args.checkpoint = checkpoint
    return checkpoint