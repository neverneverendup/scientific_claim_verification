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
from my_models import ArgBertForSequenceClassification
from argparse import ArgumentParser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device "{device}"')


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, role=None):
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
    for val in df[['id', 'claim', 'sentence', 'label', 'role']].values:
        #print(val)
        examples.append(InputExample(guid=val[0], text_a=val[2], text_b=val[1], label=val[3], role=val[4]))
    return examples

role_type = ['OBJECTIVE', 'BACKGROUND', "METHODS", "RESULTS", "CONCLUSIONS", 'none']


def convert_examples_to_features(examples, tokenizer, max_seq_length, split_num,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for example_index, example in enumerate(examples):

        context_tokens = tokenizer.tokenize(example.text_a)
        ending_tokens = tokenizer.tokenize(example.text_b)

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

        #print(example.guid, choices_features, label, one_hot_role)

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

def get_evi_role_dict():
    evidence_role = jsonlines.open('../../data/para_scifact/scifact_all_evidence_with_role.jsonl', 'r')
    evi_role = {}
    for line in evidence_role:
        evi_role[line["id"]] = line["roles"]
    return evi_role

evi_role = get_evi_role_dict()

def read_examples_by_abstract(claim, abstract, roles):
    examples = []
    for i, sen in enumerate(abstract):
        #print(i, claim, sen, roles[i])
        examples.append(InputExample(guid=i, text_a=sen, text_b=claim, label=None, role=roles[i]))
    return examples

def predict(model, tokenizer, args):
    print('predicting...')
    dataset = jsonlines.open(args.dataset)
    abstract_retrieval = jsonlines.open(args.abstract_retrieval)

    inference_labels = []
    gold_labels = []
    inference_logits = []
    corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus)}
    results = []
    n = 0
    for data, retrieval in tqdm(list(zip(dataset, abstract_retrieval))):
        n += 1
        # if n == 50 :
        #     break
        #print(data['id'] ,retrieval['claim_id'])
        assert data['id'] == retrieval['claim_id']
        claim = data['claim']

        evidences = {}
        for doc_id in retrieval['doc_ids']:
            doc = corpus[doc_id]
            sentences = doc['abstract']

            eval_examples = read_examples_by_abstract(claim, sentences, evi_role[int(doc_id)])
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length,
                                                         args.split_num, False)
            input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long).to(device)
            input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long).to(device)
            segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long).to(device)
            arg_feature = torch.tensor([f.role for f in eval_features], dtype=torch.float32).to(device)
            #label_ids = torch.tensor([f.label for f in eval_features], dtype=torch.long).to(device)

            with torch.no_grad():
                 logits = model(input_ids=input_ids, attention_mask=input_mask, labels=None,arg_feature=arg_feature)

            logits = logits.detach().cpu().numpy()
            #inference_labels = logits[:, 0].argsort()[-5:][::-1].tolist()
            inference_labels = (logits[:, 0]>= 0.5).nonzero()[0].tolist()
            print(inference_labels)
            #print((logits[:, 0]>= 0.5).nonzero())
            evidences[doc_id] = inference_labels
        results.append({
            'claim_id': retrieval['claim_id'],
            'evidence': evidences
        })
    print(results)
    output = jsonlines.open(args.output_path, 'w')
    for result in results:
        output.write({
                'claim_id': result['claim_id'],
                'evidence': result['evidence']
            })

def main():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", default='../data/random_state_24/data_origin_0', type=str, required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model", default='roberta-large', type=str, required=False,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--output_dir", default='../model/roberta_large_5121/roberta_large_5121_0', type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--corpus", default='../../data/para_scifact/corpus.jsonl', type=str)
    parser.add_argument("--dataset", default='../../data/para_scifact/claims_dev.jsonl', type=str)
    parser.add_argument("--abstract_retrieval", default='abstract_retrieval.jsonl', type=str)
    parser.add_argument("--checkpoint", default='model/argfeature/pytorch_model.bin', type=str)
    parser.add_argument("--output_path", default='argfeature_predicted_rationale_selection.jsonl', type=str)

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

    model = ArgBertForSequenceClassification(rep_file="roberta-large", args=args)
    model.load_state_dict(torch.load(args.checkpoint))
    model = model.to(device)

    print('model loaded..')
    #model = None
    predict(model, tokenizer, args)


if __name__ == '__main__':
    main()