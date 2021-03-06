# -*- coding: utf-8 -*
# @Time    : 2021/11/26 15:23
# @Author  : gzy
# @File    : evaluate_pipeline.py

from evaluation.evaluation_model import merge_rationale_label
import numpy as np
import random
import argparse
import torch
import os
import jsonlines


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate SciFact predictions.'
    )
    # dataset parameters.
    parser.add_argument('--corpus_path', type=str, default='../data/corpus.jsonl',
                        help='The corpus of documents.')
    parser.add_argument('--claim_train_path', type=str,
                        default='../data/claims_train_retrieved.jsonl')
    parser.add_argument('--claim_dev_path', type=str,
                        default='../data/claims_dev_retrieved.jsonl')
    parser.add_argument('--claim_test_path', type=str,
                        default='../data/claims_dev_retrieved.jsonl')
    parser.add_argument('--gold', type=str, default='../data/claims_dev.jsonl')
    parser.add_argument('--abstract_retrieval', type=str,
                        default='prediction/abstract_retrieval.jsonl')
    parser.add_argument('--rationale_selection', type=str,
                        default='prediction/rationale_selection.jsonl')
    parser.add_argument('--save', type=str, default='model/',
                        help='Folder to save the weights')
    parser.add_argument('--output_dir', type=str, default='./model/biobert_joint_model',
                        help='Folder to save the weights')
    parser.add_argument('--output_label', type=str, default='prediction/label_predictions.jsonl')
    parser.add_argument('--merge_results', type=str, default='prediction/merged_predictions.jsonl')
    parser.add_argument('--output', type=str, default='prediction/result_evaluation.json',
                        help='The predictions.')
    parser.add_argument('--pre_trained_model', type=str)
    parser.add_argument('--checkpoint', type=str)

    # model parameters.
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--threshold', type=float, default=0.5, required=False)
    parser.add_argument('--only_rationale', action='store_true')
    parser.add_argument('--batch_size_gpu', type=int, default=8,
                        help='The batch size to send through GPU')
    parser.add_argument('--batch-size-accumulated', type=int, default=256,
                        help='The batch size for each gradient update')
    parser.add_argument('--bert-lr', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--mode', type=str, default='claim_and_rationale',
                        choices=['claim_and_rationale', 'only_claim', 'only_rationale'])
    parser.add_argument('--filter', type=str, default='structured',
                        choices=['structured', 'unstructured'])

    parser.add_argument("--hidden_dim", type=int, default=1024,
                        help="Hidden dimension")
    parser.add_argument('--vocab_size', type=int, default=31116)
    parser.add_argument("--dropout", type=float, default=0.5, help="drop rate")
    parser.add_argument('--k', type=int, default=10, help="number of abstract retrieval(training)")
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--lambdas', type=float, default=[1, 2, 12])
    parser.add_argument('--state', type=str, default='train', choices=['train', 'prediction'])

    return parser.parse_args()


def printf(args, split):
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    if split:
        print('split: True')
    else:
        print('split: False')
    print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-')


def main():
    seed = 12345
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # loader dataset
    split = False
    eval_prediction = True

    if split:
        # split_dataset('../data/claims_train_retrieval.jsonl')
        claim_train_path = '../data/train_data_Bio.jsonl'
        claim_dev_path = '../data/dev_data_Bio.jsonl'
        claim_test_path = '../data/claims_dev_retrieved.jsonl'
        # print(claim_test_path)
    else:
        claim_train_path = args.claim_train_path
        claim_dev_path = args.claim_dev_path
        claim_test_path = args.claim_dev_path

    # args.model = 'allenai/scibert_scivocab_cased'
    # args.model = 'model/SciBert_checkpoint'
    # args.pre_trained_model = 'model/pre-train.model'
    args.model = 'dmis-lab/biobert-large-cased-v1.1-mnli'
    # args.model = 'roberta-large'
    args.epochs = 2
    args.bert_lr = 1e-5
    args.lr = 5e-6
    args.batch_size_gpu = 8
    args.dropout = 0
    args.k = 30
    k_train = 12
    args.hidden_dim = 1024  # 768/1024
    args.alpha = 1.9  # BioBert-large
    # args.alpha = 2.2  # RoBerta-large
    # args.alpha = 6.7  # BioBert-large share

    # args.lambdas = [0.2, 1.1, 12.0]  # BioBert-large w   /get
    # args.lambdas = [0.9, 2.6, 11.1]  # RoBerta-large w
    args.lambdas = [0.1, 4.7, 10.8]  # BioBert-large w/o /get
    # args.lambdas = [2.7, 2.2, 11.7]  # RoBerta-large w/o
    # args.lambdas = [1.6, 2.5, 9.5]  # # BioBert-large share w

    printf(args, split)
    rationales = [line for line in jsonlines.open(args.rationale_selection)]
    print(rationales)
    labels = [line for line in jsonlines.open(args.output_label)]

    merge_rationale_label(rationales, labels, args, state='valid', gold=args.gold)



if __name__ == '__main__':
    main()

