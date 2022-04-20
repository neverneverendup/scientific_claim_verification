import argparse
import torch
import jsonlines
import random
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset, SequentialSampler, RandomSampler
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score ,accuracy_score
import numpy as np
from my_models import TestBertForSequenceClassification, TestBertGRUForSequenceClassification,TestBertGRUForSequenceClassification2, TestBertLast2EmdForSequenceClassification
from argparse import ArgumentParser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device "{device}"')

def get_multiple_fact_condition_tuples():
    d = []
    for line in jsonlines.open('claims_train_mimo.jsonl','r'):
        if "stmt 1" in line["statements"].keys():
            d.append(line["statements"]["stmt 1"])
        else:
            d.append(None)
    print(d)
    fact = []
    condition = []
    for i in d:
        print(i)
        if i != None:
            ss = ''
            for j in i['fact tuples']:
                s = ''
                for jj in j:
                    if jj!='NIL':
                        s += jj + ' '
                print(s)
                print(j)
                if s !='':
                    s+= '. '
                ss += s
            fact.append(ss)
            css = ''
            for j in i['condition tuples']:
                s = ''
                for jj in j:
                    if jj!='NIL':
                        s += jj + ' '
                print(s)
                print(j)
                if s !='':
                    s+= '. '
                css += s
            condition.append(css)

        else:
            fact.append('')
            condition.append('')

    print(len(fact), len(condition))
    print(fact)
    return fact, condition

def get_single_fact_condition_tuples(file='train'):
    d = []
    for line in jsonlines.open('claims_'+file+'_mimo.jsonl','r'):
        if "stmt 1" in line["statements"].keys():
            d.append(line["statements"]["stmt 1"])
        else:
            d.append(None)

    fact = []
    fact_d = {}
    condition_d = {}
    condition = []
    for i in d:
        #print(i)
        if i != None:
            ss = ''
            for j in i['fact tuples'][:1]:
                s = ''
                for jj in j:
                    if jj!='NIL':
                        s += jj + ' '
                #print(s)
                #print(j)
                if s !='':
                    s+= '. '
                ss += s
            fact.append(ss)
            fact_d[i['text'][:-2] + '.'] = s

            css = ''
            for j in i['condition tuples']:
                s = ''
                for jj in j:
                    if jj!='NIL':
                        s += jj + ' '
                # print(s)
                # print(j)
                if s !='':
                    s+= '. '
                css += s
            condition.append(css)
            condition_d[i['text'][:-2] + '.'] = s
        else:
            fact.append('')
            condition.append('')

    print(len(fact), len(condition))
    return fact_d, condition_d

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, fact, text_a, condition='', text_b=None, label=None, role=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.fact = fact
        self.condition = condition
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.role = role

class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label,
                 role
                 ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label
        self.role = role

# df = pd.read_csv('../data/random_state_24/data_origin_0/train.csv')
# label_id2cate = dict(enumerate(df.categories.unique()))
# label_cate2id = {value: key for key, value in label_id2cate.items()}
# print(label_cate2id)

label_cate2id = { True : 0, False : 1 }

def read_examples(input_file, is_training):
    df = pd.read_csv(input_file)
    df['label'] = df['evidence'].map(label_cate2id)

    # exm_num =0
    # if 'train' in input_file:
    #     exm_num = 10000
    # else:
    #     exm_num = 1000

    examples = []
    facts_d, conditions_d = get_single_fact_condition_tuples(input_file.split('.')[0])

    for id, val in enumerate(df[['id', 'claim', 'sentence', 'label', 'role']].values):
        if val[1] in facts_d.keys():
            #print('ok')
            examples.append(InputExample(fact=facts_d[val[1]], condition=conditions_d[val[1]],  guid=val[0], text_a=val[2], text_b=val[1], label=val[3], role=val[4]))
        else:
            #print('false')
            examples.append(InputExample(fact='',  guid=val[0], text_a=val[2], text_b=val[1], label=val[3], role=val[4]))

    return examples

role_type = ['OBJECTIVE', 'BACKGROUND', "METHODS", "RESULTS", "CONCLUSIONS", 'none']

def convert_examples_to_features(examples, tokenizer, max_seq_length, split_num,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for example_index, example in enumerate(examples):
        #print(example.text_b +' '+ example.fact + example.condition)
        # 检验一下 拼接成果
        context_tokens = tokenizer.tokenize(example.text_a)
        ending_tokens = tokenizer.tokenize(example.text_b +' ' + example.fact + ' ' + example.condition)

        skip_len = len(context_tokens) / split_num
        choices_features = []
        for i in range(split_num):
            context_tokens_choice = context_tokens[int(i * skip_len):int((i + 1) * skip_len)]
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)
            tokens = ["<s>"] + ending_tokens + ["</s>"] + context_tokens_choice + ["</s>"]
            segment_ids = [0] * (len(ending_tokens) + 2) + [1] * (len(context_tokens_choice) + 1)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding_length = max_seq_length - len(input_ids)
            input_ids += ([0] * padding_length)
            input_mask += ([0] * padding_length)
            segment_ids += ([0] * padding_length)
            choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        role = example.role
        one_hot_role = ([0] * len(role_type))
        one_hot_role[role_type.index(role)] = 1

        features.append(
            InputFeatures(
                example_id=example.guid,
                choices_features=choices_features,
                label=label,
                role=one_hot_role
            )
        )
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        else:
            tokens_a.pop()

        # if len(tokens_a) > len(tokens_b):
        #     tokens_a.pop()
        # else:
        #     tokens_b.pop()

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

def accuracy(out, labels):
    #print(out)
    #print(labels)
    #outputs = np.argmax(out, axis=1)
    #return f1_score(labels,outputs,labels=[0,1,2],average='macro')
    #print(accuracy_score(labels,out))
    #print(f1_score(labels,out,labels=[0,1],average='macro'))
    f1 = f1_score(labels, out, zero_division=0)
    precision = precision_score(labels, out, zero_division=0)
    recall = recall_score(labels, out, zero_division=0)
    return f1, precision, recall

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def evaluate(model, file, tokenizer, args):
    inference_labels = []
    gold_labels = []
    inference_logits = []
    eval_examples = read_examples(file, is_training=True)
    eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length,
                                                 args.split_num, False)
    all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
    all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)


    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        # input_ids = input_ids.view(-1,input_ids.size(-1))
        # input_mask = input_mask.view(-1,input_mask.size(-1))

        with torch.no_grad():
            loss, logits  = model(input_ids=input_ids, attention_mask=input_mask, labels=label_ids)
            # logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
            # logits = outputs.logits
            # loss = outputs.loss
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        inference_labels.extend(np.argmax(logits, axis=1).tolist())
        gold_labels.append(label_ids)
        inference_logits.append(logits)
        eval_loss += loss.mean().item()
        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    gold_labels = np.concatenate(gold_labels, 0)
    inference_logits = np.concatenate(inference_logits, 0)
    model.train()
    eval_loss = eval_loss / nb_eval_steps
    #print(eval_loss, eval_accuracy)
    return accuracy(inference_labels, gold_labels)

def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default='../data/random_state_24/data_origin_0', type=str, required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model", default='roberta-large', type=str, required=False,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--output_dir", default='../model/roberta_large_5121/roberta_large_5121_0', type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--batch_size_gpu", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument('--batch_size_accumulated', type=int, default=64,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    # parser.add_argument("--learning_rate", default=5e-5, type=float,
    #                     help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--lstm_hidden_size", default=512, type=int,
                        help="")
    parser.add_argument("--lstm_layers", default=1, type=int,
                        help="")
    parser.add_argument("--lstm_dropout", default=0.1, type=float,
                        help="")

    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--split_num", default=1, type=int,
                        help="text split")

    parser.add_argument("--num_labels", default=2, type=float,
                        help="not_do_eval_steps.")
    parser.add_argument("--bert_hidden_size", default=1024, type=float,
                        help="not_do_eval_steps.")
    parser.add_argument("--epochs", default=20, type=float,
                        help="not_do_eval_steps.")
    parser.add_argument("--bert_lr", default=1e-5, type=float,
                        help="not_do_eval_steps.")
    parser.add_argument("--lr", default=1e-3, type=float,
                        help="not_do_eval_steps.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model, num_labels=2)
    model = TestBertForSequenceClassification(rep_file="roberta-large", args=args)

    optimizer = torch.optim.Adam([
        # If you are using non-roberta based models, change this to point to the right base
        {'params': model.bert.parameters(), 'lr': args.bert_lr},
        {'params': model.classifier.parameters(), 'lr': args.lr}
    ])
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 20)

    # d = []
    # for line in jsonlines.open('claims_train_mimo.jsonl'):
    #     d.append(line["statements"]["stmt 1"]["text"])
    train_examples = read_examples('train.csv', is_training = True)
    train_features = convert_examples_to_features(
        train_examples, tokenizer, args.max_seq_length,args.split_num, True)

    all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
    all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
    model.to(device)
    best_acc = 0.0
    for e in range(args.epochs):
        model.train()
        t = tqdm(DataLoader(train_data, batch_size=args.batch_size_gpu, shuffle=True))
        for i, batch in enumerate(t):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            # input_ids = input_ids.view(-1, input_ids.size(-1))
            # input_mask = input_mask.view(-1,input_mask.size(-1))
            loss, _ = model(input_ids=input_ids,  attention_mask=input_mask,
                            labels=label_ids)
            # loss = outputs.loss
            loss.backward()
            if (i + 1) % (args.batch_size_accumulated // args.batch_size_gpu) == 0:
                optimizer.step()
                optimizer.zero_grad()
                t.set_description(f'Epoch {e}, iter {i}, loss: {round(loss.item(), 4)}')

        scheduler.step()
        # Eval
        train_score = evaluate(model, 'train.csv', tokenizer, args)
        print(f'Epoch {e}, train f1: %.4f, precision: %.4f, recall: %.4f' % train_score)
        dev_score = evaluate(model, 'dev.csv', tokenizer, args)
        print(f'Epoch {e}, dev f1: %.4f, precision: %.4f, recall: %.4f' % dev_score)

        if dev_score[0] > best_acc :
            print("=" * 80)
            print("Best F1", dev_score[0])
            print("Saving Model......")
            best_acc = dev_score[0]
            # Save a trained model
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Only save the model it-self
            output_model_file = os.path.join('model/mimo/', "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            print("=" * 80)
        else:
            print("=" * 80)
















if __name__ == '__main__':
    main()