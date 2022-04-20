# -*- coding: utf-8 -*
# @Time    : 2021/11/20 17:18
# @Author  : gzy
# @File    : test_mimo_output.py
import jsonlines
import pandas as pd

def get_multiple_fact_condition_tuples():
    d = []
    for line in jsonlines.open('claims_train_mimo.jsonl','r'):
        if "stmt 1" in line["statements"].keys():
            d.append(line["statements"]["stmt 1"])
        else:
            d.append(None)
    print(d)
    fact = []
    fact_d = {}
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
            fact_d[i['text']] = ss
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
    #print(fact)
    print(fact_d)
    return fact, condition

def get_single_fact_condition_tuples(file='train'):
    d = []
    for line in jsonlines.open('claims_'+file+'_mimo.jsonl','r'):
        if "stmt 1" in line["statements"].keys():
            d.append(line["statements"]["stmt 1"])
        else:
            d.append(None)
    #print(d)
    fact = []
    fact_d = {}
    condition = []
    for i in d:
        print(i)
        if i != None:
            ss = ''
            for j in i['fact tuples'][:1]:
                s = ''
                for jj in j:
                    if jj!='NIL':
                        s += jj + ' '

                if s !='':
                    s+= '. '
                ss += s
            fact.append(ss)
            fact_d[i['text'][:-2] + '.'] = s

            css = ''

            condition.append(css)

        else:
            fact.append('')
            condition.append('')

    print(len(fact), len(condition))
    #print(fact)
    #print(fact_d)
    return fact_d

def prepare_train_data_with_mimo(input_file='dev.csv'):
    df = pd.read_csv(input_file)
    facts_d = get_single_fact_condition_tuples(input_file.split('.')[0])
    for id, val in enumerate(df.values):
        if val[1] in facts_d.keys():
            print('ok')
        else:
            print('false')


if __name__ == '__main__':

    #get_single_fact_condition_tuples()
    prepare_train_data_with_mimo()




