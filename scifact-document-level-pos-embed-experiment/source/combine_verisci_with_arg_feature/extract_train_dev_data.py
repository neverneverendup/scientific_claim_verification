# -*- coding: utf-8 -*
# @Time    : 2021/10/4 10:53
# @Author  : gzy
# @File    : extract_train_dev_data.py

import argparse
import torch
import jsonlines

from torch.utils.data import Dataset, DataLoader
import pandas

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str,  default="../../data/para_scifact/corpus.jsonl")
parser.add_argument('--claim-train', type=str, default="../../data/para_scifact/claims_train_biosent_retrieved.jsonl")
parser.add_argument('--claim-dev', type=str, default="../../data/para_scifact/claims_dev_biosent_retrieved.jsonl")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device "{device}"')

def get_evi_role_dict():
    evidence_role = jsonlines.open('../../data/para_scifact/scifact_all_evidence_with_role.jsonl', 'r')
    evi_role = {}
    for line in evidence_role:
        evi_role[line["id"]] = line["roles"]
    return evi_role

evi_role = get_evi_role_dict()
class SciFactRationaleSelectionDataset(Dataset):
    def __init__(self, corpus: str, claims: str):
        self.samples = []
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
        for claim in jsonlines.open(claims):
            # 便利当前claim的evidence项，如果evidence是空的就跳过了
            for doc_id, evidence in claim['evidence'].items():
                doc = corpus[int(doc_id)]
                # 这里直接把每个rationale中的句子都提取出来作为evidence sentence了，不受rationale范围限制
                evidence_sentence_idx = {s for es in evidence for s in es['sentences']}
                # 这里，evidence abstract的句子里面，属于evidence sentence的句子label是1，不属于的label作为负样本是0
                # 另外注意这里最后组合的训练数据，脱离了abstract的范畴，每条数据就是独立的句对关系判断数据了
                # 左边是claim,右边是abstract sentence, 最后是label可以看成代表两个句子是否entailment
                for i, sentence in enumerate(doc['abstract']):
                    self.samples.append({
                        'claim': claim['claim'],
                        'sentence': sentence,
                        'evidence': i in evidence_sentence_idx,
                        'role' : evi_role[int(doc_id)][i]
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == '__main__':
    trainset = SciFactRationaleSelectionDataset(args.corpus, args.claim_train)
    devset = SciFactRationaleSelectionDataset(args.corpus, args.claim_dev)

    for idx, set in enumerate([trainset, devset]):
        claims = []
        sentences = []
        evidences = []
        roles = []

        for i in set.samples:
            print(i)
            claims.append(i['claim'])
            sentences.append(i['sentence'])
            evidences.append(i['evidence'])
            roles.append(i['role'])

        d = {'claim':claims, 'sentence':sentences, 'evidence':evidences, 'role':roles}
        df = pandas.DataFrame(d)
        # if idx ==0 :
        #     df.to_csv('train.csv')
        # else:
        #     df.to_csv('dev.csv')

        label_cate2id = {True: 0, False: 1}
        print(df['evidence'])
        df['label'] = df['evidence'].map(label_cate2id)
        print(df['label'])




