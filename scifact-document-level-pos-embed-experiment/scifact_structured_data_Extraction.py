# -*- coding: utf-8 -*
# @Time    : 2021/9/11 10:25
# @Author  : gzy
# @File    : scifact_structured_data_Extraction.py

import os
import jsonlines
import re
import random

# 类别别名映射字典
d = {}
d['BACKGROUND'] = ['CONTEXT','BACKGROUND', 'INTRODUCTION','BACKGROUND AND PURPOSE', 'BACKGROUND AND AIMS','INTRODUCTION','RECENT FINDINGS']
d['OBJECTIVE'] = ['OBJECTIVE', 'OBJECTIVES', 'PURPOSE','AIMS','AIM','SUBJECTS']
d['CONCLUSIONS'] = ['CONCLUSIONS', 'CONCLUSION', 'INTERPRETATION','CONCLUSIONS AND RELEVANCE','CONCLUSIONS AND SIGNIFICANCE']
d['METHODS'] = ['METHODS','SUBJECTS' ,'DESIGN','PARTICIPANTS', 'MAIN OUTCOME MEASURES', 'MAIN OUTCOME MEASURE','SETTING','DESIGN AND SETTING','DESIGN AND METHODS', 'RESEARCH DESIGN AND METHODS','EXPERIMENTAL DESIGN','METHOD','PATIENTS AND METHODS','INTERVENTION','INTERVENTIONS','MATERIALS AND METHODS','MAIN OUTCOMES AND MEASURES']
# 'METHODS AND RESULTS' METHODS AND FINDINGS 需要人工干预标注, 后者已经标注完毕，前者需要继续标注
d['RESULTS'] = ['RESULTS', 'MAIN RESULTS', 'MEASUREMENTS AND MAIN RESULTS','FINDINGS']

def extract_structured_data(file='corpus-annotated.jsonl'):
    structured_data = []
    categories = []
    with jsonlines.open(file,'r') as f:
        for line in f:
            if line['structured'] == True:
                structured_data.append(line)
    print(len(structured_data))

    for data in structured_data:
        for sen in data['abstract']:
            rs = re.match('^([A-Z]{3,} ){1,5}', sen)
            if rs:
                categories.append(rs.group().strip())

    category_count = {}
    for i in categories:
        category_count[i] = category_count.get(i,0)+1
    #print(category_count)
    #print(sorted(categories))
    print(len(set(categories)))
    print(sorted(category_count.items(), key = lambda kv:kv[1], reverse=True))

def organize_sequence_labeling_data(file='corpus-annotated.json'):
    structured_data = []
    categories = []
    output = open('sequence_labeling_data_recategory_filtered.txt','w',encoding='utf-8')

    with jsonlines.open(file,'r') as f:
        for line in f:
            if line['structured'] == True:
                structured_data.append(line)

    num = 0
    for data in structured_data:
        temp_data = []
        f = False
        for sen in data['abstract']:
            flag = False
            print(sen)
            sen = sen.strip()
            rs = re.match('^([A-Z]{3,} ){1,5}', sen)
            if rs:
                # 如果匹配上，则直到下次匹配上之前后面几句都是这个类别
                cat = rs.group().strip()
                ans = sen.split(cat)
                for k in d:
                    if cat in d[k]:
                        cat = k
                        flag = True
                        break
                if not flag:
                    # 包含类别体系外的标签，丢弃数据
                    f = True
                    break
                print(ans)
                #print(ans[-1].strip(),'\t',cat, file=output)
                temp_data.append(ans[-1].strip()+'\t'+cat)
            else:
                temp_data.append(sen+'\t'+cat)
                #print(sen,'\t',cat, file=output)
        if f:
            continue
        for t in temp_data:
            print(t, file=output)
        num += 1
        print('', file=output)
    print(num)
    print(len(structured_data))


def train_test_split(test_ration=0.1):
    output = open('sequence_labeling_data_recategory_filtered.txt','r',encoding='utf-8')
    train_file = open('scifact_train.txt','w',encoding='utf-8')
    test_file = open('scifact_test.txt','w',encoding='utf-8')

    paragraphs = []
    t = []
    for line in output:
        #print(line)
        if line.strip()!='':
            t.append(line.strip())
        else:
            paragraphs.append(t)
            t = []
    print(paragraphs)
    print(len(paragraphs))
    test_ids = random.sample(range(len(paragraphs)), int(len(paragraphs)*0.1))
    print(test_ids)

    for i in range(len(paragraphs)):
        if i in test_ids:
            for sen in paragraphs[i]:
                print(sen, file=test_file)
            print('', file=test_file)
        else:
            for sen in paragraphs[i]:
                print(sen, file=train_file)
            print('', file=train_file)

if __name__ == '__main__':
    #extract_structured_data()
    #organize_sequence_labeling_data()
    train_test_split()