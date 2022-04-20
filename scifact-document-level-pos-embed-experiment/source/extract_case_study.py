# -*- coding: utf-8 -*
# @Time    : 2021/9/20 22:12
# @Author  : gzy
# @File    : extract_case_study.py
import jsonlines

def get_evi_role_dict():
    evidence_role = jsonlines.open('../data/para_scifact/scifact_all_evidence_with_role.jsonl', 'r')
    #evidence_role = jsonlines.open('../../data/para_scifact/update_scifact_all_evidence_with_role.jsonl', 'r')
    evi_role = {}
    for line in evidence_role:
        evi_role[line["id"]] = line["roles"]
    return evi_role

def extract_examples(file='compare_different_rs_train_data/model/arg_model_k0_use_ref/rationale_selection.jsonl'):
    evi_role = get_evi_role_dict()

    arg_preds = []
    for p in jsonlines.open('compare_different_rs_train_data/model/author_model_k0_use_ref/rationale_selection.jsonl'):
        arg_preds.append(p)


    claims = []
    for claim in jsonlines.open('../data/para_scifact/claims_dev_tfidf_retrieved.jsonl'):
        claims.append(claim)
    #print(claims)

    with jsonlines.open(file) as f:
        for line in f:
            for c in claims:
                if c['id'] == line['claim_id']:
                    print(c, line)
                    for doc_id in list(c['evidence'].keys()):
                        flag1 = False
                        flag2 = False
                        evidence = c['evidence'][doc_id]
                        evidence_sentence_idx = [s for es in evidence for s in es['sentences']]
                        print(doc_id,'ans', evidence_sentence_idx,'pred', line['evidence'][doc_id])
                        if line['evidence'][doc_id] == evidence_sentence_idx:
                            #print('hit!')
                            flag1 = True
                        for p in arg_preds:
                            if p['claim_id'] == c['id']:
                                print(doc_id, 'ans', evidence_sentence_idx, 'arg pred', p['evidence'][doc_id])

                                if p['evidence'][doc_id] != line['evidence'][doc_id]:
                                    print('diff!')
                                    flag2 = True

                        # if not flag1 and flag2:
                        #     print('done')
                                    print(evi_role[int(doc_id)])


def extract_abstract_examples():
    structured_data = {}
    with jsonlines.open('../data/para_scifact/corpus.jsonl', 'r') as f:
        for line in f:
            if line['structured'] == True:
                structured_data[line['doc_id']] = line

    evi_role = get_evi_role_dict()
    claims = []
    for claim in jsonlines.open('../data/para_scifact/claims_dev_tfidf_retrieved.jsonl'):
        claims.append(claim)

    for c in claims:
        print(c)
        for doc_id in list(c['evidence'].keys()):
            evidence = c['evidence'][doc_id]
            for es in evidence:
                # and int(doc_id) in structured_data.keys():
                if len(es['sentences']) > 1 :
                    print(es['sentences'])
                    print(doc_id, evi_role[int(doc_id)])


if __name__ == '__main__':
    #extract_examples()
    extract_abstract_examples()
    '''
    1568684 ans [1, 3, 4] pred [3, 4]
2 hit!
done
['BACKGROUND', 'BACKGROUND', 'BACKGROUND', 'RESULTS', 'RESULTS', 'CONCLUSIONS', 'CONCLUSIONS']

22180793 ans [4, 7] pred [4]
2 hit!
done
['BACKGROUND', 'BACKGROUND', 'RESULTS', 'RESULTS', 'RESULTS', 'RESULTS', 'CONCLUSIONS', 'CONCLUSIONS']
    
52873726 ans [3] pred []
2 hit!
done
['BACKGROUND', 'RESULTS', 'RESULTS', 'RESULTS', 'RESULTS', 'RESULTS', 'RESULTS', 'RESULTS', 'CONCLUSIONS']

    
9650982 ans [3] pred []
2 hit!
done
['OBJECTIVE', 'METHODS', 'RESULTS', 'RESULTS', 'CONCLUSIONS']

{'id': 275, 'claim': 'Combining phosphatidylinositide 3-kinase and MEK 1/2 inhibitors is effective at treating KRAS mutant tumors.', 'evidence': {'4961038': [{'sentences': [7], 'label': 'SUPPORT'}, {'sentences': [8], 'label': 'SUPPORT'}], '14241418': [{'sentences': [10], 'label': 'SUPPORT'}]}, 'cited_doc_ids': [4961038, 14241418, 14819804], 'retrieved_doc_ids': ['4961038', '4920376', '17462437', '85665741', '36310858', '4320424', '24190159', '2272614', '3210545', '41650417', '11200685', '7317051', '3360428', '1285713', '28651643', '27270151', '3559136', '21150010', '19752008', '4702639', '18682109', '26491450', '5389095', '5849439', '7821634', '27240699', '18956141', '4959368', '9955779', '30919024']} {'claim_id': 275, 'evidence': {'4961038': [7], '14241418': []}}
4961038 ans [7, 8] pred [7]
4961038 ans [7, 8] arg pred [7, 8]
diff!
['BACKGROUND', 'BACKGROUND', 'BACKGROUND', 'BACKGROUND', 'METHODS', 'RESULTS', 'RESULTS', 'RESULTS', 'CONCLUSIONS']


    
    '''