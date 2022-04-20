# -*- coding: utf-8 -*
# @Time    : 2021/8/18 17:47
# @Author  : gzy
# @File    : get_document_max_length.py

import jsonlines
corpus = '../data/para_scifact/corpus.jsonl'
corpus =  [ doc for doc in jsonlines.open(corpus)]
max_len = 0
x = [86217760]
for c in corpus:
    #print(c)

    #print(len(c["abstract"]))

    if c["doc_id"] in x:
        continue

    if len(c["abstract"]) > 300:
        continue

    if len(c["abstract"]) > max_len:
        print(len(c["abstract"]))
        max_len = len(c["abstract"])

print(max_len)

