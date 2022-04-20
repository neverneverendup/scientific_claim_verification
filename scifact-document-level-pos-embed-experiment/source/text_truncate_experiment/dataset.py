import random
import jsonlines
from torch.utils.data import Dataset
from util import read_passages, clean_words, test_f1, to_BIO, from_BIO, clean_url, clean_num
from transformers import AutoTokenizer
import torch

def get_evi_role_dict():
    evidence_role = jsonlines.open('../../data/para_scifact/scifact_all_evidence_with_role.jsonl', 'r')
    #evidence_role = jsonlines.open('../../data/para_scifact/update_scifact_all_evidence_with_role.jsonl', 'r')

    evi_role = {}
    for line in evidence_role:
        evi_role[line["id"]] = line["roles"]
    return evi_role

tokenizer = AutoTokenizer.from_pretrained('roberta-large')
evi_role = get_evi_role_dict()

def list_to_string(l):
    s = ''
    for i in l:
        s = s + str(l) + '_'
    return s

def string_to_list(s):
    s = s.split('_')
    data = []
    for i in s:
        if i == '_' or i =='':
            continue
        data.append(i.replace('_',''))
    return data

''' TRAIN
'dataset': 0,  # 表明是人造数据
            'claim': claim['claim'],
            'claim_id': claim['id'],
            'doc_id': doc['doc_id'],
            'paragraph': concat_sentences,
            'label': rationale_label_string,
            'stance': self.stance_ind[stance],
            'sentence_ids': kept_sentences_ids
'''

''' TEST
'dataset': 1,
'claim': claim['claim'],
'claim_id': claim['id'],
'doc_id': doc['doc_id'],
'paragraph': concat_sentences,
'''

def my_collcate_fn(batch):

    dataset_list = [item['dataset'] for item in batch]
    claim_list = [item['claim'] for item in batch]
    claimid_list = [item['claim_id'] for item in batch]
    docid_list = [item['doc_id'] for item in batch]
    paragraph_list = [item['paragraph'] for item in batch]
    label_list = []
    stance_list = []
    sentenceids_list = []
    role_list = []

    if 'label' in batch[0].keys():
        label_list = [item['label'] for item in batch]
    if 'stance' in batch[0].keys():
        stance_labels = [item['stance'] for item in batch]
        stance_list = torch.tensor(stance_labels, dtype=torch.long)
    if 'sentence_ids' in batch[0].keys():
        sentenceids_list = [item['sentence_ids'] for item in batch]
    if 'role' in batch[0].keys():
        role_list = [item['role'] for item in batch]


    return {
        'dataset': dataset_list,
        'claim': claim_list,
        'claim_id': claimid_list,
        'doc_id': docid_list,
        'paragraph': paragraph_list,
        'label': label_list,
        'stance': stance_list,
        'sentence_ids': sentenceids_list,
        'role':role_list
    }



# 如果长度过长丢弃METHOD(OBJECTIVE)句子
class TruncateArgSciFactParagraphBatchDataset(Dataset):
    """
    Dataset for a feeding a paragraph to a single BERT model.
    """
    def __init__(self, corpus: str, claims: str, discard_role_classes, sep_token="</s>", k=0, train=True, dummy=True,
                 downsample_n=0, downsample_p=0.5):
        self.label_ind = {"NEI": 0, "rationale": 1}
        self.rev_label_ind = {i: l for (l, i) in self.label_ind.items()}
        self.stance_ind = {"NEI": 0, "SUPPORT": 1, "CONTRADICT": 2}
        self.rev_stance_ind = {i: l for (l, i) in self.stance_ind.items()}

        self.samples = []
        self.excluded_pairs = []
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
        i = 0
        for claim in jsonlines.open(claims):
            # i+=1
            # if i > 3:
            #     break
            if k > 0 and "retrieved_doc_ids" in claim:
                candidates = claim["retrieved_doc_ids"][:k]
            else:
                candidates = claim["cited_doc_ids"]
                #candidates = []
            candidates = [int(cand) for cand in candidates]
            if train:
                evidence_doc_ids = [int(ID) for ID in list(claim['evidence'].keys())]
                all_candidates = sorted(list(set(candidates + evidence_doc_ids)))
            else:
                all_candidates = candidates

            for doc_id in all_candidates:
                doc = corpus[int(doc_id)]
                doc_id = str(doc_id)

                if "discourse" in doc:
                    abstract_sentences = \
                        [discourse + " " + sentence.strip() for discourse, sentence in
                         zip(doc['discourse'], doc['abstract'])]
                else:
                    abstract_sentences = [sent.strip() for sent in doc['abstract']]

                inputs = claim["claim"] + ' ' + sep_token + ' ' + (" " + sep_token + " ").join(abstract_sentences)
                encoded_dict = tokenizer.encode_plus(inputs)
                # 如果文本长度过长，丢弃METHOD句子
                roles = evi_role[doc['doc_id']]
                #print(encoded_dict)
                #print(len(encoded_dict['input_ids']))
                if len(encoded_dict['input_ids']) > 512:
                    method_sen_ids = [i for i,val in enumerate(roles) if val in discard_role_classes]
                    #print(roles)
                    #print(method_sen_ids)
                else:
                    method_sen_ids = []
                    #continue
                this_role = [val for i,val in enumerate(roles) if i not in method_sen_ids]
                sen_ids = [i for i,val in enumerate(roles) if i not in method_sen_ids]

                if train:
                    for down_n in range(downsample_n + 1):
                        if doc_id in claim['evidence']:
                            evidence = claim['evidence'][doc_id]
                            evidence_sentence_idx = {s for es in evidence for s in es['sentences']}
                            stances = set([es["label"] for es in evidence])

                            if "SUPPORT" in stances:
                                stance = "SUPPORT"
                            elif "CONTRADICT" in stances:
                                stance = "CONTRADICT"
                            else:
                                stance = "NEI"

                            if down_n > 0:
                                abstract_sentences, evidence_sentence_idx, stance, kept_sentences_ids, method_sen_ids = \
                                    self.downsample(abstract_sentences, evidence_sentence_idx, stance, downsample_p, roles)
                                method_sen_ids = []
                                t = []
                                sen_ids = []

                                for i in kept_sentences_ids.split('_'):
                                    if i == '_' or i =='':
                                        continue
                                    sen_ids.append(int(i.replace('_','')))

                                for i,val in enumerate(roles):
                                    if i in sen_ids:
                                        t.append(val)
                                this_role = t

                                if len(abstract_sentences) == 0:
                                    break

                        else:
                            evidence_sentence_idx = {}
                            stance = "NEI"

                        #concat_sentences = (" " + sep_token + " ").join(abstract_sentences)
                        concat_sentences = (" " + sep_token + " ").join([val for i,val in enumerate(abstract_sentences) if i not in method_sen_ids])
                        #print(concat_sentences)
                        concat_sentences = clean_num(clean_url(concat_sentences))
                        rationale_label_string = "".join(["1" if i in evidence_sentence_idx else "0" for i in \
                                                          range(len(abstract_sentences))])
                        rationale_label_string = "".join([val for i,val in enumerate(rationale_label_string) if i not in method_sen_ids])

                        if dummy:
                            concat_sentences = "@ " + sep_token + " " + concat_sentences
                            rationale_label_string = "0" + rationale_label_string
                            this_role = ['none'] + this_role

                        kept_sentences_ids = []
                        if down_n > 0:
                            self.samples.append({
                                'dataset': 0,  # 表明是人造数据
                                'claim': claim['claim'],
                                'claim_id': claim['id'],
                                'doc_id': doc['doc_id'],
                                'paragraph': concat_sentences,
                                'label': rationale_label_string,
                                'stance': self.stance_ind[stance],
                                'sentence_ids': sen_ids,
                                'role':this_role
                            })
                        else:
                            self.samples.append({
                                'dataset': 1,
                                'claim': claim['claim'],
                                'claim_id': claim['id'],
                                'doc_id': doc['doc_id'],
                                'paragraph': concat_sentences,
                                'label': rationale_label_string,
                                'stance': self.stance_ind[stance],
                                'sentence_ids': sen_ids,
                                'role': this_role

                            })

                        if doc_id not in claim['evidence']:
                            break  # Do not downsample if contain no evidence
                else:
                    concat_sentences = (" " + sep_token + " ").join(
                        [val for i, val in enumerate(abstract_sentences) if i not in method_sen_ids])

                    concat_sentences = clean_num(clean_url(concat_sentences))
                    if dummy:
                        concat_sentences = "@ " + sep_token + " " + concat_sentences
                        this_role = ['none'] + this_role

                    self.samples.append({
                        'dataset': 1,
                        'claim': claim['claim'],
                        'claim_id': claim['id'],
                        'doc_id': doc['doc_id'],
                        'paragraph': concat_sentences,
                        'role': this_role,
                        'sentence_ids': sen_ids

                    })
                    # 这个下采样 是根据一篇摘要的内容，从它中随机挑选一些句子当做一个fake abstract，如果挑选的句子是原来的 rationale label，就保持

    # 如果采样到了rationale sentence，则stance label保持和原abstract相同
    def downsample(self, abstract_sentences, evidence_sentence_idx, stance, downsample_p, roles):
        kept_sentences = []
        kept_sentences_ids = ""
        evidence_bitmap = []
        for i, sentence in enumerate(abstract_sentences):
            if random.random() < downsample_p:
                kept_sentences.append(sentence)
                kept_sentences_ids += str(i) + '_'
                if i in evidence_sentence_idx:
                    evidence_bitmap.append(True)
                else:
                    evidence_bitmap.append(False)
        kept_evidence_idx = []
        for i, e in enumerate(evidence_bitmap):
            if e:
                kept_evidence_idx.append(i)

        med_sent_ids = []
        for i,val in enumerate(kept_sentences_ids.split('_')):
            val = val.replace('_','')
            if val=='':
                continue
            if roles[int(val)]=='METHODS':
                med_sent_ids.append(i)

        kept_stance = stance if len(kept_evidence_idx) > 0 else "NEI"
        return kept_sentences, set(kept_evidence_idx), kept_stance, kept_sentences_ids, med_sent_ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



class NotTruncateArgSciFactParagraphBatchDataset(Dataset):
    """
    Dataset for a feeding a paragraph to a single BERT model.
    """
    def __init__(self, corpus: str, claims: str, sep_token="</s>", k=0, train=True, dummy=True,
                 downsample_n=0, downsample_p=0.5):
        self.label_ind = {"NEI": 0, "rationale": 1}
        self.rev_label_ind = {i: l for (l, i) in self.label_ind.items()}
        self.stance_ind = {"NEI": 0, "SUPPORT": 1, "CONTRADICT": 2}
        self.rev_stance_ind = {i: l for (l, i) in self.stance_ind.items()}

        self.samples = []
        self.excluded_pairs = []
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
        #i = 0
        for claim in jsonlines.open(claims):
            #i+=1
            # if i > 10:
            #     break
            if k > 0 and "retrieved_doc_ids" in claim:
                candidates = claim["retrieved_doc_ids"][:k]
            else:
                candidates = claim["cited_doc_ids"]
                #candidates = []
            candidates = [int(cand) for cand in candidates]
            if train:
                evidence_doc_ids = [int(ID) for ID in list(claim['evidence'].keys())]
                all_candidates = sorted(list(set(candidates + evidence_doc_ids)))
            else:
                all_candidates = candidates

            for doc_id in all_candidates:
                doc = corpus[int(doc_id)]
                doc_id = str(doc_id)

                if "discourse" in doc:
                    abstract_sentences = \
                        [discourse + " " + sentence.strip() for discourse, sentence in
                         zip(doc['discourse'], doc['abstract'])]
                else:
                    abstract_sentences = [sent.strip() for sent in doc['abstract']]

                inputs = claim["claim"] + ' ' + sep_token + ' ' + (" " + sep_token + " ").join(abstract_sentences)
                #print(inputs)
                encoded_dict = tokenizer.encode_plus(inputs)
                # 如果文本长度过长，丢弃METHOD句子
                roles = evi_role[doc['doc_id']]
                #print(encoded_dict)
                #print(len(encoded_dict['input_ids']))
                # if len(encoded_dict['input_ids']) > 512:
                #     method_sen_ids = [i for i,val in enumerate(roles) if val == 'METHODS']
                #     #print(roles)
                #     #print(method_sen_ids)
                # else:
                #     method_sen_ids = []
                #     #continue

                method_sen_ids = []
                this_role = [val for i,val in enumerate(roles) if i not in method_sen_ids]
                sen_ids = [i for i,val in enumerate(roles) if i not in method_sen_ids]

                if train:
                    for down_n in range(downsample_n + 1):
                        if doc_id in claim['evidence']:
                            evidence = claim['evidence'][doc_id]
                            evidence_sentence_idx = {s for es in evidence for s in es['sentences']}
                            stances = set([es["label"] for es in evidence])

                            if "SUPPORT" in stances:
                                stance = "SUPPORT"
                            elif "CONTRADICT" in stances:
                                stance = "CONTRADICT"
                            else:
                                stance = "NEI"

                            if down_n > 0:
                                abstract_sentences, evidence_sentence_idx, stance, kept_sentences_ids, method_sen_ids = \
                                    self.downsample(abstract_sentences, evidence_sentence_idx, stance, downsample_p, roles)
                                method_sen_ids = []

                                t = []
                                sen_ids = []
                                #print(kept_sentences_ids)
                                for i in kept_sentences_ids.split('_'):
                                    #print(i)
                                    if i == '_' or i =='':
                                        continue
                                    #print(i)
                                    sen_ids.append(int(i.replace('_','')))

                                for i,val in enumerate(roles):
                                    if i in sen_ids:
                                        t.append(val)
                                this_role = t
                                #print(roles)
                                #print(kept_sentences_ids)
                                #print(this_role)
                                if len(abstract_sentences) == 0:
                                    break

                        else:
                            evidence_sentence_idx = {}
                            stance = "NEI"

                        #concat_sentences = (" " + sep_token + " ").join(abstract_sentences)
                        concat_sentences = (" " + sep_token + " ").join([val for i,val in enumerate(abstract_sentences) if i not in method_sen_ids])
                        #print(abstract_sentences)
                        #print(concat_sentences)

                        concat_sentences = clean_num(clean_url(concat_sentences))
                        rationale_label_string = "".join(["1" if i in evidence_sentence_idx else "0" for i in \
                                                          range(len(abstract_sentences))])
                        #print(rationale_label_string)
                        rationale_label_string = "".join([val for i,val in enumerate(rationale_label_string) if i not in method_sen_ids])
                        #print(rationale_label_string)


                        if dummy:
                            concat_sentences = "@ " + sep_token + " " + concat_sentences
                            rationale_label_string = "0" + rationale_label_string
                            this_role = ['none'] + this_role

                        kept_sentences_ids = []

                        if down_n > 0:
                            self.samples.append({
                                'dataset': 0,  # 表明是人造数据
                                'claim': claim['claim'],
                                'claim_id': claim['id'],
                                'doc_id': doc['doc_id'],
                                'paragraph': concat_sentences,
                                'label': rationale_label_string,
                                'stance': self.stance_ind[stance],
                                'sentence_ids': sen_ids,
                                'role':this_role
                            })
                        else:
                            self.samples.append({
                                'dataset': 1,
                                'claim': claim['claim'],
                                'claim_id': claim['id'],
                                'doc_id': doc['doc_id'],
                                'paragraph': concat_sentences,
                                'label': rationale_label_string,
                                'stance': self.stance_ind[stance],
                                'sentence_ids': sen_ids,
                                'role': this_role

                            })

                        if doc_id not in claim['evidence']:
                            break  # Do not downsample if contain no evidence
                else:
                    concat_sentences = (" " + sep_token + " ").join(
                        [val for i, val in enumerate(abstract_sentences) if i not in method_sen_ids])

                    concat_sentences = clean_num(clean_url(concat_sentences))
                    if dummy:
                        concat_sentences = "@ " + sep_token + " " + concat_sentences
                        this_role = ['none'] + this_role

                    self.samples.append({
                        'dataset': 1,
                        'claim': claim['claim'],
                        'claim_id': claim['id'],
                        'doc_id': doc['doc_id'],
                        'paragraph': concat_sentences,
                        'role': this_role,
                        'sentence_ids': sen_ids

                    })
                    # 这个下采样 是根据一篇摘要的内容，从它中随机挑选一些句子当做一个fake abstract，如果挑选的句子是原来的 rationale label，就保持

    # 如果采样到了rationale sentence，则stance label保持和原abstract相同
    def downsample(self, abstract_sentences, evidence_sentence_idx, stance, downsample_p, roles):
        kept_sentences = []
        kept_sentences_ids = ""
        evidence_bitmap = []
        for i, sentence in enumerate(abstract_sentences):
            if random.random() < downsample_p:
                kept_sentences.append(sentence)
                kept_sentences_ids += str(i) + '_'
                if i in evidence_sentence_idx:
                    evidence_bitmap.append(True)
                else:
                    evidence_bitmap.append(False)
        kept_evidence_idx = []
        for i, e in enumerate(evidence_bitmap):
            if e:
                kept_evidence_idx.append(i)

        med_sent_ids = []
        for i,val in enumerate(kept_sentences_ids.split('_')):
            val = val.replace('_','')
            if val=='':
                continue
            if roles[int(val)]=='METHODS':
                med_sent_ids.append(i)

        kept_stance = stance if len(kept_evidence_idx) > 0 else "NEI"
        return kept_sentences, set(kept_evidence_idx), kept_stance, kept_sentences_ids, med_sent_ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ArgSciFactParagraphBatchDataset(Dataset):
    """
    Dataset for a feeding a paragraph to a single BERT model.
    """

    def __init__(self, corpus: str, claims: str, sep_token="</s>", k=0, train=True, dummy=True,
                 downsample_n=0, downsample_p=0.5):
        self.label_ind = {"NEI": 0, "rationale": 1}
        self.rev_label_ind = {i: l for (l, i) in self.label_ind.items()}
        self.stance_ind = {"NEI": 0, "SUPPORT": 1, "CONTRADICT": 2}
        self.rev_stance_ind = {i: l for (l, i) in self.stance_ind.items()}

        self.samples = []
        self.excluded_pairs = []
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
        #i = 0
        for claim in jsonlines.open(claims):
            # i+=1
            # if i >3:
            #     break
            if k > 0 and "retrieved_doc_ids" in claim:
                candidates = claim["retrieved_doc_ids"][:k]
            else:
                candidates = claim["cited_doc_ids"]
                #candidates = []
            candidates = [int(cand) for cand in candidates]
            if train:
                evidence_doc_ids = [int(ID) for ID in list(claim['evidence'].keys())]
                all_candidates = sorted(list(set(candidates + evidence_doc_ids)))
            else:
                all_candidates = candidates

            for doc_id in all_candidates:
                doc = corpus[int(doc_id)]
                doc_id = str(doc_id)

                if "discourse" in doc:
                    abstract_sentences = \
                        [discourse + " " + sentence.strip() for discourse, sentence in
                         zip(doc['discourse'], doc['abstract'])]
                else:
                    abstract_sentences = [sent.strip() for sent in doc['abstract']]

                if train:
                    for down_n in range(downsample_n + 1):
                        if doc_id in claim['evidence']:
                            evidence = claim['evidence'][doc_id]
                            evidence_sentence_idx = {s for es in evidence for s in es['sentences']}
                            stances = set([es["label"] for es in evidence])

                            if "SUPPORT" in stances:
                                stance = "SUPPORT"
                            elif "CONTRADICT" in stances:
                                stance = "CONTRADICT"
                            else:
                                stance = "NEI"

                            if down_n > 0:
                                abstract_sentences, evidence_sentence_idx, stance, kept_sentences_ids = \
                                    self.downsample(abstract_sentences, evidence_sentence_idx, stance, downsample_p)
                                if len(abstract_sentences) == 0:
                                    break

                        else:
                            evidence_sentence_idx = {}
                            stance = "NEI"

                        concat_sentences = (" " + sep_token + " ").join(abstract_sentences)
                        concat_sentences = clean_num(clean_url(concat_sentences))
                        rationale_label_string = "".join(["1" if i in evidence_sentence_idx else "0" for i in \
                                                          range(len(abstract_sentences))])

                        if dummy:
                            concat_sentences = "@ " + sep_token + " " + concat_sentences
                            rationale_label_string = "0" + rationale_label_string

                        if down_n > 0:
                            self.samples.append({
                                'dataset': 0,  # 表明是人造数据
                                'claim': claim['claim'],
                                'claim_id': claim['id'],
                                'doc_id': doc['doc_id'],
                                'paragraph': concat_sentences,
                                'label': rationale_label_string,
                                'stance': self.stance_ind[stance],
                                'sentence_ids': kept_sentences_ids
                            })
                        else:
                            self.samples.append({
                                'dataset': 1,
                                'claim': claim['claim'],
                                'claim_id': claim['id'],
                                'doc_id': doc['doc_id'],
                                'paragraph': concat_sentences,
                                'label': rationale_label_string,
                                'stance': self.stance_ind[stance],
                                'sentence_ids': "0"

                            })

                        if doc_id not in claim['evidence']:
                            break  # Do not downsample if contain no evidence
                else:
                    concat_sentences = (" " + sep_token + " ").join(abstract_sentences)

                    if dummy:
                        concat_sentences = "@ " + sep_token + " " + concat_sentences

                    self.samples.append({
                        'dataset': 1,
                        'claim': claim['claim'],
                        'claim_id': claim['id'],
                        'doc_id': doc['doc_id'],
                        'paragraph': concat_sentences,
                    })
                    # 这个下采样 是根据一篇摘要的内容，从它中随机挑选一些句子当做一个fake abstract，如果挑选的句子是原来的 rationale label，就保持

    # 如果采样到了rationale sentence，则stance label保持和原abstract相同
    def downsample(self, abstract_sentences, evidence_sentence_idx, stance, downsample_p):
        kept_sentences = []
        kept_sentences_ids = ""
        evidence_bitmap = []
        for i, sentence in enumerate(abstract_sentences):
            if random.random() < downsample_p:
                kept_sentences.append(sentence)
                kept_sentences_ids += str(i) + '_'
                if i in evidence_sentence_idx:
                    evidence_bitmap.append(True)
                else:
                    evidence_bitmap.append(False)
        kept_evidence_idx = []
        for i, e in enumerate(evidence_bitmap):
            if e:
                kept_evidence_idx.append(i)
        kept_stance = stance if len(kept_evidence_idx) > 0 else "NEI"
        return kept_sentences, set(kept_evidence_idx), kept_stance, kept_sentences_ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# If a paragraph is longer than MAX_SEQ_LEN, treat it as an another paragraph.
class SciDTdataset(Dataset):
    def __init__(self, path: str, MAX_SEQ_LEN: int, CHUNK_SIZE:int, label_ind=None, train=False, shuffle=False, BIO=True):
        self.shuffle = shuffle
        self.n_paragraph_slices = 0
        self.MAX_SEQ_LEN = MAX_SEQ_LEN
        self.CHUNK_SIZE = CHUNK_SIZE
        n_pieces = MAX_SEQ_LEN // CHUNK_SIZE
        n_pieces += 1 if MAX_SEQ_LEN % CHUNK_SIZE > 0 else 0
        self.samples = []
        self.true_pairs = [] # The unprocessed paragraph - tag pairs.
        str_seqs, label_seqs = read_passages(path, is_labeled=train)
        self.str_seqs = str_seqs
        self.label_seqs = label_seqs
        for pi, str_seq in enumerate(str_seqs):
            self.true_pairs.append({
                'paragraph_id': pi,
                'paragraph': str_seq,
                'label': label_seqs[pi]
            })
        
        str_seqs = clean_words(str_seqs)
        if BIO:
            label_seqs = to_BIO(label_seqs)
        
        if not label_ind:
            self.label_ind = {"none": 0}
        else:
            self.label_ind = label_ind
            
        if len(self.label_ind)<=1:
            for str_seq, label_seq in zip(str_seqs, label_seqs):
                for label in label_seq:
                    if label not in self.label_ind:
                        # Add new labels with values 0,1,2,....
                        self.label_ind[label] = len(self.label_ind)
                        
        self.rev_label_ind = {i: l for (l, i) in self.label_ind.items()}
        
        for pi, str_seq in enumerate(str_seqs):
            n_paragraph_slices = len(str_seq) // MAX_SEQ_LEN
            n_paragraph_slices += 1 if len(str_seq) % MAX_SEQ_LEN > 0 else 0
            self.n_paragraph_slices += n_paragraph_slices
            for p_slice in range(n_paragraph_slices):
                this_slice = str_seq[p_slice*MAX_SEQ_LEN : (p_slice+1) * MAX_SEQ_LEN]
                padded_paragraph = this_slice + ["" for i in range(CHUNK_SIZE * n_pieces - len(this_slice))]
                
                if train:
                    this_slice_tag = label_seqs[pi][p_slice*MAX_SEQ_LEN : (p_slice+1) * MAX_SEQ_LEN]
                    padded_tag = this_slice_tag + ["none" for i in range(CHUNK_SIZE * n_pieces - len(this_slice))]
                
                for p in range(n_pieces):
                    this_piece = padded_paragraph[p*CHUNK_SIZE: (p+1)*CHUNK_SIZE]
                    if train:
                        this_piece_tag = padded_tag[p*CHUNK_SIZE: (p+1)*CHUNK_SIZE]
                        
                    for i, sentence in enumerate(this_piece):
                        sentence_id = i + p*CHUNK_SIZE + p_slice * MAX_SEQ_LEN
                        this_sample = {
                            'paragraph_id': pi,
                            'sentence': sentence,
                            'sentence_id': sentence_id, 
                        }
                        
                        if train:
                            this_sample['label'] = self.label_ind[this_piece_tag[i]]
                        
                        self.samples.append(this_sample)

    def __make_shuffle_idx(self):
        self.paragraph_indices = [i for i in range(self.n_paragraph_slices)]
        random.shuffle(self.paragraph_indices)
                        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.shuffle: 
            if idx == 0:
                self.__make_shuffle_idx()

            paragraph_idx = idx // self.MAX_SEQ_LEN
            offset = idx % self.MAX_SEQ_LEN

            original_idx = self.paragraph_indices[paragraph_idx] * self.MAX_SEQ_LEN + offset
            
        else:
            original_idx = idx
        
        return self.samples[original_idx]
    
# If a paragraph is longer than MAX_SEQ_LEN, treat it as an another paragraph.
class SciFactSubParagraphDataset(Dataset):
    def __init__(self, corpus: str, claims: str, MAX_SEQ_LEN: int, CHUNK_SIZE:int, train=False, shuffle=False, negative_paragraph_sample_ratio = 1, negative_sentence_sample_ratio = 1):
        
        
        def sample_negative_sentence(sentences, rationale_labels, negative_paragraph_sample_ratio):
            kept_sentences = []
            kept_labels = []
            while len(kept_sentences) == 0: # Avoid empty sentences returned
                for i, sentence in enumerate(sentences):
                    if i in rationale_labels or random.random() < negative_paragraph_sample_ratio:
                        kept_sentences.append(sentence)
                        kept_labels.append(i in rationale_labels)
            return kept_sentences, kept_labels
                
                
        
        self.shuffle = shuffle
        self.n_paragraph_slices = 0
        self.MAX_SEQ_LEN = MAX_SEQ_LEN
        self.CHUNK_SIZE = CHUNK_SIZE
        n_pieces = MAX_SEQ_LEN // CHUNK_SIZE
        n_pieces += 1 if MAX_SEQ_LEN % CHUNK_SIZE > 0 else 0
        self.label_ind = {"NEI": 0, "rationale": 1}
        self.rev_label_ind = {i: l for (l, i) in self.label_ind.items()}
        self.stance_ind = {"NEI": 0, "SUPPORT": 1, "CONTRADICT": 2}
        self.rev_stance_ind = {i: l for (l, i) in self.stance_ind.items()}
        
        self.samples = []
        self.true_pairs = [] # The unprocessed claim - abstract pairs.
        self.excluded_pairs = []
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
        
        for claim in jsonlines.open(claims):
            for doc_id in claim["cited_doc_ids"]:
                doc = corpus[int(doc_id)]
                doc_id = str(doc_id)
                if doc_id in claim['evidence']:
                    evidence = claim['evidence'][doc_id]
                    evidence_sentence_idx = {s for es in evidence for s in es['sentences']}
                    stances = set([es["label"] for es in evidence])
                    still_include = False
                    if "SUPPORT" in stances:
                        stance = "SUPPORT"
                    elif "CONTRADICT" in stances:
                        stance = "CONTRADICT"
                    else:
                        stance = "NEI"
                else: 
                    evidence_sentence_idx = {}
                    stance = "NEI"
                    still_include = random.random() < negative_paragraph_sample_ratio
                
                if stance != "NEI" or still_include:
                    sentences, labels = sample_negative_sentence(doc['abstract'], evidence_sentence_idx, 
                                                                 negative_paragraph_sample_ratio)
                    
                    self.true_pairs.append({
                        'dataset': 1,
                        'claim': claim['claim'],
                        'claim_id': claim['id'],
                        'doc_id': doc['doc_id'],
                        'paragraph': sentences,
                        'label': labels,
                        'stance': stance
                    })
                    
                    
                    
                    n_paragraph_slices = len(sentences) // MAX_SEQ_LEN
                    n_paragraph_slices += 1 if len(sentences) % MAX_SEQ_LEN > 0 else 0
                    self.n_paragraph_slices += n_paragraph_slices
                    
                    for p_slice in range(n_paragraph_slices):
                        this_slice = sentences[p_slice*MAX_SEQ_LEN : (p_slice+1) * MAX_SEQ_LEN]
                        padded_paragraph = this_slice + ["" for i in range(CHUNK_SIZE * n_pieces - len(this_slice))]
                        for p in range(n_pieces):
                            this_piece = padded_paragraph[p*CHUNK_SIZE: (p+1)*CHUNK_SIZE]
                            for i, sentence in enumerate(this_piece):
                                sentence_id = i + p*CHUNK_SIZE + p_slice * MAX_SEQ_LEN
                                if len(sentence) > 0:
                                    label = 1 if sentence_id in evidence_sentence_idx else 0
                                    mask = 1
                                    sentence_stance = self.stance_ind[stance] if label == 1 else self.stance_ind["NEI"]
                                else:
                                    label = 0
                                    mask = 0
                                    sentence_stance = self.stance_ind["NEI"]
                                self.samples.append({
                                    'dataset': 1,
                                    'claim': claim['claim'],
                                    'claim_id': claim['id'],
                                    'sentence': sentence,
                                    'doc_id': doc['doc_id'],
                                    'sentence_id': sentence_id, 
                                    'label': label,
                                    'sentence_stance': sentence_stance,
                                    'stance': self.stance_ind[stance],
                                    'mask': mask
                                })
                                           
                else:
                    self.excluded_pairs.append({
                        'dataset': 1,
                        'claim': claim['claim'],
                        'claim_id': claim['id'],
                        'doc_id': doc['doc_id'],
                        'paragraph': doc['abstract'],
                        'label': [1 if i in evidence_sentence_idx else 0 for i in range(len(doc['abstract']))],
                        'stance': stance
                    })

    def __make_shuffle_idx(self):
        self.paragraph_indices = [i for i in range(self.n_paragraph_slices)]
        random.shuffle(self.paragraph_indices)
                        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.shuffle: 
            if idx == 0:
                self.__make_shuffle_idx()

            paragraph_idx = idx // self.MAX_SEQ_LEN
            offset = idx % self.MAX_SEQ_LEN

            original_idx = self.paragraph_indices[paragraph_idx] * self.MAX_SEQ_LEN + offset
            
        else:
            original_idx = idx
        
        return self.samples[original_idx]


class SciFactStancePredictionDataset(Dataset):
    """
    Dataset for a taking the predicted rationale and predict stance.
    """

    def __init__(self, corpus: str, claims: str, rationales: str, sep_token="</s>"):
        self.label_ind = {"NEI": 0, "rationale": 1}
        self.rev_label_ind = {i: l for (l, i) in self.label_ind.items()}
        self.stance_ind = {"NEI": 0, "SUPPORT": 1, "CONTRADICT": 2}
        self.rev_stance_ind = {i: l for (l, i) in self.stance_ind.items()}

        self.samples = []
        self.excluded_pairs = []
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}

        for claim, rationale in zip(jsonlines.open(claims), jsonlines.open(rationales)):
            N_rationale = sum([len(v) for k, v in rationale["evidence"].items()])
            if N_rationale > 0:
                for doc_id in rationale["evidence"]:
                    doc = corpus[int(doc_id)]
                    doc_id = str(doc_id)
                    evidence_sentence_idx = rationale["evidence"][doc_id]
                    if len(evidence_sentence_idx) > 0:
                        selected_sentences = []
                        for i, sentence in enumerate(doc['abstract']):
                            if i in evidence_sentence_idx:
                                selected_sentences.append(sentence)

                        concat_sentences = (" " + sep_token + " ").join(selected_sentences)
                        concat_sentences = clean_num(clean_url(concat_sentences))

                        self.samples.append({
                            'dataset': 1,
                            'claim': claim['claim'],
                            'claim_id': claim['id'],
                            'doc_id': doc['doc_id'],
                            'paragraph': concat_sentences
                        })
            else:
                self.excluded_pairs.append({
                    'dataset': 1,
                    'claim': claim['claim'],
                    'claim_id': claim['id']
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class SciFactParagraphDataset(Dataset):
    """
    Dataset for a feeding a paragraph to a single BERT model.
    """
    def __init__(self, corpus: str, claims: str, train=False, negative_paragraph_sample_ratio = 1, negative_sentence_sample_ratio = 1, N_sample = 1):
        
        def sample_negative_sentence(sentences, rationale_labels, negative_paragraph_sample_ratio):
            kept_sentences = []
            kept_labels = []
            while len(kept_sentences) == 0: # Avoid empty sentences returned
                for i, sentence in enumerate(sentences):
                    if i in rationale_labels or random.random() < negative_paragraph_sample_ratio:
                        kept_sentences.append(sentence)
                        kept_labels.append(i in rationale_labels)
            return kept_sentences, kept_labels
        
        self.label_ind = {"NEI": 0, "rationale": 1}
        self.rev_label_ind = {i: l for (l, i) in self.label_ind.items()}
        self.stance_ind = {"NEI": 0, "SUPPORT": 1, "CONTRADICT": 2}
        self.rev_stance_ind = {i: l for (l, i) in self.stance_ind.items()}
        
        self.samples = []
        self.excluded_pairs = []
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
        
        for N in range(N_sample):
            for claim in jsonlines.open(claims):
                for doc_id in claim["cited_doc_ids"]:
                    doc = corpus[int(doc_id)]
                    doc_id = str(doc_id)
                    
                    if "discourse" in doc:
                        abstract_sentences = \
                        [discourse + " " + sentence for discourse, sentence in zip(doc['discourse'], doc['abstract'])]
                    else:
                        abstract_sentences = doc['abstract']
                    
                    if doc_id in claim['evidence']:
                        evidence = claim['evidence'][doc_id]
                        evidence_sentence_idx = {s for es in evidence for s in es['sentences']}
                        stances = set([es["label"] for es in evidence])
                        still_include = False
                        if "SUPPORT" in stances:
                            stance = "SUPPORT"
                        elif "CONTRADICT" in stances:
                            stance = "CONTRADICT"
                        else:
                            stance = "NEI"
                    else: 
                        evidence_sentence_idx = {}
                        stance = "NEI"
                        still_include = random.random() < negative_paragraph_sample_ratio

                    if stance != "NEI" or still_include:
                        selected_sentences, selected_labels = sample_negative_sentence(
                            abstract_sentences, evidence_sentence_idx, negative_sentence_sample_ratio)
                        

                        self.samples.append({
                            'dataset': 1,
                            'claim': claim['claim'],
                            'claim_id': claim['id'],
                            'doc_id': doc['doc_id'],
                            'paragraph': selected_sentences,
                            'label': selected_labels,
                            'stance': self.stance_ind[stance]
                        })                        

                    else:
                        self.excluded_pairs.append({
                            'dataset': 1,
                            'claim': claim['claim'],
                            'claim_id': claim['id'],
                            'doc_id': doc['doc_id'],
                            'paragraph': abstract_sentences,
                            'label': [1 if i in evidence_sentence_idx else 0 for i in range(len(doc['abstract']))],
                            'stance': self.stance_ind[stance]
                        })
                        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
class SciFactStancePredictionDataset(Dataset):
    """
    Dataset for a taking the predicted rationale and predict stance.
    """
    def __init__(self, corpus: str, claims: str, rationales: str, sep_token="</s>"):
        self.label_ind = {"NEI": 0, "rationale": 1}
        self.rev_label_ind = {i: l for (l, i) in self.label_ind.items()}
        self.stance_ind = {"NEI": 0, "SUPPORT": 1, "CONTRADICT": 2}
        self.rev_stance_ind = {i: l for (l, i) in self.stance_ind.items()}
        
        self.samples = []
        self.excluded_pairs = []
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
        
        for claim, rationale in zip(jsonlines.open(claims), jsonlines.open(rationales)):
            N_rationale = sum([len(v) for k, v in rationale["evidence"].items()])
            if N_rationale > 0:
                for doc_id in rationale["evidence"]:
                    doc = corpus[int(doc_id)]
                    doc_id = str(doc_id)
                    evidence_sentence_idx = rationale["evidence"][doc_id]
                    if len(evidence_sentence_idx)>0:
                        selected_sentences = []
                        for i, sentence in enumerate(doc['abstract']):
                            if i in evidence_sentence_idx:
                                selected_sentences.append(sentence)

                        concat_sentences = (" "+sep_token+" ").join(selected_sentences)
                        concat_sentences = clean_num(clean_url(concat_sentences))

                        self.samples.append({
                            'dataset': 1,
                            'claim': claim['claim'],
                            'claim_id': claim['id'],
                            'doc_id': doc['doc_id'],
                            'paragraph': concat_sentences
                        })
            else:
                self.excluded_pairs.append({
                    'dataset': 1,
                    'claim': claim['claim'],
                    'claim_id': claim['id']
                })
                    
                        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
class FEVERParagraphDataset(Dataset):
    """
    Dataset for a feeding a paragraph to a single BERT model.
    """
    def __init__(self, data_path):
        self.label_ind = {"NEI": 0, "rationale": 1}
        self.rev_label_ind = {i: l for (l, i) in self.label_ind.items()}
        self.stance_ind = {"NOT ENOUGH INFO": 0, "SUPPORTS": 1, "REFUTES": 2}
        self.rev_stance_ind = {i: l for (l, i) in self.stance_ind.items()}
        
        self.samples = []
        self.nei_pairs = []
        
        for data in jsonlines.open(data_path):
            if len(data["sentences"]) > 0:
                rationales = []
                for evid in data["evidence_sets"]:
                    rationales.extend(evid)
                evidence_idx = set(rationales)
                self.samples.append({
                    'dataset': 0,
                    'claim': data['claim'],
                    'claim_id': data['id'],
                    'paragraph': data["sentences"],
                    'label': [1 if i in evidence_idx else 0 for i in range(len(data["sentences"]))],
                    'stance': self.stance_ind[data["label"]]
                })                        

            else:
                self.nei_pairs.append({
                    'dataset': 0,
                    'claim': data['claim'],
                    'claim_id': data['id']
                })
                        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
class SciFact_FEVER_Dataset(Dataset):
    def __init__(self, dataset1, dataset2, multiplier = 1):
        if len(dataset1) < len(dataset2):
            self.samples = dataset1.samples * multiplier + dataset2.samples
        elif len(dataset1) > len(dataset2):
            self.samples = dataset1.samples + dataset2.samples * multiplier
        else:
            self.samples = dataset1.samples + dataset2.samples
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
class Multiple_SciFact_Dataset(Dataset):
    def __init__(self, dataset, multiplier = 1):
        self.samples = dataset.samples * multiplier
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class SciFactParagraphBatchDataset(Dataset):
    """
    Dataset for a feeding a paragraph to a single BERT model.
    """

    def __init__(self, corpus: str, claims: str, sep_token="</s>", k=0, train=True, dummy=True,
                 downsample_n=0, downsample_p=0.5, evaluation=False, use_ref_abstract=False):
        self.label_ind = {"NEI": 0, "rationale": 1}
        self.rev_label_ind = {i: l for (l, i) in self.label_ind.items()}
        self.stance_ind = {"NEI": 0, "SUPPORT": 1, "CONTRADICT": 2}
        self.rev_stance_ind = {i: l for (l, i) in self.stance_ind.items()}

        self.samples = []
        self.excluded_pairs = []
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}

        if evaluation:
            print('evaluation dataset')
        else:
            print('train dataset')
        if use_ref_abstract:
            print('use reference abstract')
        if k > 0 :
            print('use {} retrieved_docs'.format(k))

        for claim in jsonlines.open(claims):
            candidates = []

            if k > 0 and "retrieved_doc_ids" in claim :
                candidates.extend(claim["retrieved_doc_ids"][:k])
            if use_ref_abstract:
                candidates.extend(claim["cited_doc_ids"])

            candidates = [int(cand) for cand in candidates]

            # else:
            #     #candidates = claim["cited_doc_ids"]
            #     candidates = []

            if train:
                evidence_doc_ids = [int(ID) for ID in list(claim['evidence'].keys())]
                all_candidates = sorted(list(set(candidates + evidence_doc_ids)))
                # 这里注意evaluation的时候调整
                # all_candidates = sorted(list(set(candidates)))
            else:
                evidence_doc_ids = [int(ID) for ID in list(claim['evidence'].keys())]
                all_candidates = sorted(list(set(candidates + evidence_doc_ids)))


            if evaluation:
                candidates = claim["cited_doc_ids"]
                evidence_doc_ids = [int(ID) for ID in list(claim['evidence'].keys())]
                all_candidates = sorted(list(set(candidates + evidence_doc_ids)))

            # all_candidates = []
            # evidence_doc_ids = [int(ID) for ID in list(claim['evidence'].keys())]
            # all_candidates = sorted(list(set(evidence_doc_ids)))

            for doc_id in all_candidates:
                doc = corpus[int(doc_id)]
                doc_id = str(doc_id)

                if "discourse" in doc:
                    abstract_sentences = \
                        [discourse + " " + sentence.strip() for discourse, sentence in
                         zip(doc['discourse'], doc['abstract'])]
                else:
                    abstract_sentences = [sent.strip() for sent in doc['abstract']]

                if train:
                    for down_n in range(downsample_n + 1):
                        if doc_id in claim['evidence']:
                            evidence = claim['evidence'][doc_id]
                            evidence_sentence_idx = {s for es in evidence for s in es['sentences']}
                            stances = set([es["label"] for es in evidence])

                            if "SUPPORT" in stances:
                                stance = "SUPPORT"
                            elif "CONTRADICT" in stances:
                                stance = "CONTRADICT"
                            else:
                                stance = "NEI"

                            if down_n > 0:
                                abstract_sentences, evidence_sentence_idx, stance = \
                                    self.downsample(abstract_sentences, evidence_sentence_idx, stance, downsample_p)
                                if len(abstract_sentences) == 0:
                                    break

                        else:
                            evidence_sentence_idx = {}
                            stance = "NEI"

                        concat_sentences = (" " + sep_token + " ").join(abstract_sentences)
                        concat_sentences = clean_num(clean_url(concat_sentences))
                        rationale_label_string = "".join(["1" if i in evidence_sentence_idx else "0" for i in \
                                                          range(len(abstract_sentences))])

                        if dummy:
                            concat_sentences = "@ " + sep_token + " " + concat_sentences
                            rationale_label_string = "0" + rationale_label_string

                        self.samples.append({
                            'dataset': 1,
                            'claim': claim['claim'],
                            'claim_id': claim['id'],
                            'doc_id': doc['doc_id'],
                            'paragraph': concat_sentences,
                            'label': rationale_label_string,
                            'stance': self.stance_ind[stance]
                        })

                        if doc_id not in claim['evidence']:
                            break  # Do not downsample if contain no evidence
                else:
                    concat_sentences = (" " + sep_token + " ").join(abstract_sentences)

                    if dummy:
                        concat_sentences = "@ " + sep_token + " " + concat_sentences

                    self.samples.append({
                        'dataset': 1,
                        'claim': claim['claim'],
                        'claim_id': claim['id'],
                        'doc_id': doc['doc_id'],
                        'paragraph': concat_sentences,
                    })
                    # 这个下采�?是根据一篇摘要的内容，从它中随机挑选一些句子当做一个fake abstract，如果挑选的句子是原来的 rationale label，就保持

    # 如果采样到了rationale sentence，则stance label保持和原abstract相同
    def downsample(self, abstract_sentences, evidence_sentence_idx, stance, downsample_p):
        kept_sentences = []
        evidence_bitmap = []
        for i, sentence in enumerate(abstract_sentences):
            if random.random() < downsample_p:
                kept_sentences.append(sentence)
                if i in evidence_sentence_idx:
                    evidence_bitmap.append(True)
                else:
                    evidence_bitmap.append(False)
        kept_evidence_idx = []
        for i, e in enumerate(evidence_bitmap):
            if e:
                kept_evidence_idx.append(i)
        kept_stance = stance if len(kept_evidence_idx) > 0 else "NEI"
        return kept_sentences, set(kept_evidence_idx), kept_stance

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# class SciFactParagraphBatchDataset(Dataset):
#     """
#     Dataset for a feeding a paragraph to a single BERT model.
#     """
#     def __init__(self, corpus: str, claims: str, sep_token="</s>", k=0, train = True, eval=False, dummy=True,
#                  downsample_n = 0, downsample_p = 0.5):
#         self.label_ind = {"NEI": 0, "rationale": 1}
#         self.rev_label_ind = {i: l for (l, i) in self.label_ind.items()}
#         self.stance_ind = {"NEI": 0, "SUPPORT": 1, "CONTRADICT": 2}
#         self.rev_stance_ind = {i: l for (l, i) in self.stance_ind.items()}
#
#         self.samples = []
#         self.excluded_pairs = []
#         corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
#         max_length = 0
#         for claim in jsonlines.open(claims):
#             if k > 0 and "retrieved_doc_ids" in claim:
#                 candidates = claim["retrieved_doc_ids"][:k]
#             else:
#                 candidates = claim["cited_doc_ids"]
#                 #candidates = []
#
#             candidates = [int(cand) for cand in candidates]
#             # if train:
#             #     evidence_doc_ids = [int(ID) for ID in list(claim['evidence'].keys())]
#             #     all_candidates = sorted(list(set(candidates + evidence_doc_ids)))
#             # else:
#             #     all_candidates = candidates
#
#             #if eval:
#             all_candidates = []
#             evidence_doc_ids = [int(ID) for ID in list(claim['evidence'].keys())]
#             all_candidates = sorted(list(set(evidence_doc_ids)))
#
#             for doc_id in all_candidates:
#
#                 doc = corpus[int(doc_id)]
#                 doc_id = str(doc_id)
#
#                 if len(doc["abstract"]) > max_length:
#                     max_length = len(doc["abstract"])
#
#                 if "discourse" in doc:
#                     abstract_sentences = \
#                     [discourse + " " + sentence.strip() for discourse, sentence in zip(doc['discourse'], doc['abstract'])]
#                 else:
#                     abstract_sentences = [sent.strip() for sent in doc['abstract']]
#
#                 if train:
#                     for down_n in range(downsample_n+1):
#                         if doc_id in claim['evidence']:
#                             evidence = claim['evidence'][doc_id]
#                             evidence_sentence_idx = {s for es in evidence for s in es['sentences']}
#                             stances = set([es["label"] for es in evidence])
#
#                             if "SUPPORT" in stances:
#                                 stance = "SUPPORT"
#                             elif "CONTRADICT" in stances:
#                                 stance = "CONTRADICT"
#                             else:
#                                 stance = "NEI"
#
#                             if down_n > 0:
#                                 abstract_sentences, evidence_sentence_idx, stance, kept_sentences_ids = \
#                                     self.downsample(abstract_sentences, evidence_sentence_idx, stance, downsample_p)
#                                 if len(abstract_sentences) == 0:
#                                     break
#
#                         else:
#                             evidence_sentence_idx = {}
#                             stance = "NEI"
#
#                         concat_sentences = (" "+sep_token+" ").join(abstract_sentences)
#                         concat_sentences = clean_num(clean_url(concat_sentences))
#                         rationale_label_string = "".join(["1" if i in evidence_sentence_idx else "0" for i in \
#                                                           range(len(abstract_sentences))])
#
#                         if dummy:
#                             concat_sentences = "@ "+sep_token+" "+concat_sentences
#                             rationale_label_string = "0"+rationale_label_string
#
#                         if down_n > 0:
#                             self.samples.append({
#                                 'dataset': 0, #表明是人造数据
#                                 'claim': claim['claim'],
#                                 'claim_id': claim['id'],
#                                 'doc_id': doc['doc_id'],
#                                 'paragraph': concat_sentences,
#                                 'label': rationale_label_string,
#                                 'stance': self.stance_ind[stance],
#                                 'sentence_ids': kept_sentences_ids
#                             })
#                         else:
#                             self.samples.append({
#                                 'dataset': 1,
#                                 'claim': claim['claim'],
#                                 'claim_id': claim['id'],
#                                 'doc_id': doc['doc_id'],
#                                 'paragraph': concat_sentences,
#                                 'label': rationale_label_string,
#                                 'stance': self.stance_ind[stance],
#                                 'sentence_ids': "0"
#                             })
#
#                         if doc_id not in claim['evidence']:
#                             break # Do not downsample if contain no evidence
#                 else:
#                     concat_sentences = (" "+sep_token+" ").join(abstract_sentences)
#
#                     if dummy:
#                         concat_sentences = "@ "+sep_token+" "+concat_sentences
#
#                     self.samples.append({
#                         'dataset': 1,
#                         'claim': claim['claim'],
#                         'claim_id': claim['id'],
#                         'doc_id': doc['doc_id'],
#                         'paragraph': concat_sentences,
#                     })
#
#         print(max_length)
#     # 这个下采样 是根据一篇摘要的内容，从它中随机挑选一些句子当做一个fake abstract，如果挑选的句子是原来的 rationale label，就保持
#     # 如果采样到了rationale sentence，则stance label保持和原abstract相同
#     def downsample(self, abstract_sentences, evidence_sentence_idx, stance, downsample_p):
#         kept_sentences = []
#         kept_sentences_ids = ""
#         evidence_bitmap = []
#         for i, sentence in enumerate(abstract_sentences):
#             if random.random() < downsample_p:
#                 kept_sentences.append(sentence)
#                 kept_sentences_ids += str(i)+'_'
#                 if i in evidence_sentence_idx:
#                     evidence_bitmap.append(True)
#                 else:
#                     evidence_bitmap.append(False)
#         kept_evidence_idx = []
#         for i, e in enumerate(evidence_bitmap):
#             if e:
#                 kept_evidence_idx.append(i)
#         kept_stance = stance if len(kept_evidence_idx) > 0 else "NEI"
#         return kept_sentences, set(kept_evidence_idx), kept_stance, kept_sentences_ids
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         return self.samples[idx]
    
class FEVERParagraphBatchDataset(Dataset):
    """
    Dataset for a feeding a paragraph to a single BERT model.
    """
    def __init__(self, datapath: str, sep_token="</s>", train = True, k = 0, dummy=True):
        self.label_ind = {"NEI": 0, "rationale": 1}
        self.rev_label_ind = {i: l for (l, i) in self.label_ind.items()}
        self.stance_ind = {"NOT ENOUGH INFO": 0, "SUPPORTS": 1, "REFUTES": 2}
        self.rev_stance_ind = {i: l for (l, i) in self.stance_ind.items()}
        
        self.samples = []
        self.excluded_pairs = []
        
        def max_sent_len(sentences):
            return max([len(sent.split()) for sent in sentences])
        
        for data in jsonlines.open(datapath):
            try:
                if len(data["sentences"]) > 0:
                    sentences = data["sentences"]
                    if max_sent_len(sentences) > 100 or len(sentences) > 100:
                        continue
                    concat_sentences = (" "+sep_token+" ").join(sentences)
                    concat_sentences = clean_num(clean_url(concat_sentences))
                    if train:
                        rationales = []
                        for evid in data["evidence_sets"]:
                            rationales.extend(evid)
                        evidence_idx = set(rationales)
                        rationale_label_string = "".join(["1" if i in evidence_idx else "0" for i in range(len(sentences))])
                        
                        if dummy:
                            concat_sentences = "@ "+sep_token+" "+concat_sentences
                            rationale_label_string = "0"+rationale_label_string 
                        
                        self.samples.append({
                            'dataset': 0,
                            'claim': data['claim'],
                            'claim_id': data['id'],
                            'paragraph': concat_sentences,
                            'label': rationale_label_string,
                            'stance': self.stance_ind[data["label"]]
                        })
                    elif data["hit"]: # The retrieved pages hit the gold page.
                        if dummy:
                            concat_sentences = "@ "+sep_token+" "+concat_sentences
                        self.samples.append({
                            'dataset': 0,
                            'claim': data['claim'],
                            'claim_id': data['id'],
                            'paragraph': concat_sentences
                        })
            except:
                pass
            try:
                if len(data["negative_sentences"]) > 0:
                    for sentences in data["negative_sentences"][:k]:
                        if max_sent_len(sentences) > 100 or len(sentences) > 100:
                            continue
                        
                        concat_sentences = (" "+sep_token+" ").join(sentences)
                        concat_sentences = clean_num(clean_url(concat_sentences))

                        if train:
                            rationale_label_string = "0" * len(sentences)
                            
                            if dummy:
                                concat_sentences = "@ "+sep_token+" "+concat_sentences
                                rationale_label_string = "0"+rationale_label_string 
                            
                            self.samples.append({
                                'dataset': 0,
                                'claim': data['claim'],
                                'claim_id': data['id'],
                                'paragraph': concat_sentences,
                                'label': rationale_label_string,
                                'stance': self.stance_ind["NOT ENOUGH INFO"]
                            })
                        else:
                            if dummy:
                                concat_sentences = "@ "+sep_token+" "+concat_sentences
                                
                            self.samples.append({
                                'dataset': 0,
                                'claim': data['claim'],
                                'claim_id': data['id'],
                                'paragraph': concat_sentences
                            })
            except:
                pass
                    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
class SciFactStanceDataset(Dataset):
    """
    Dataset for a feeding a paragraph to a single BERT model.
    """
    def __init__(self, corpus: str, claims: str, sep_token="</s>", k=0, train = True):
        self.label_ind = {"NEI": 0, "rationale": 1}
        self.rev_label_ind = {i: l for (l, i) in self.label_ind.items()}
        self.stance_ind = {"NEI": 0, "SUPPORT": 1, "CONTRADICT": 2}
        self.rev_stance_ind = {i: l for (l, i) in self.stance_ind.items()}
        
        self.samples = []
        self.excluded_pairs = []
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
        
        for claim in jsonlines.open(claims):
            if k > 0 and "retrieved_doc_ids" in claim:
                candidates = claim["retrieved_doc_ids"][:k]
            else:
                candidates = claim["cited_doc_ids"]
                
            candidates = [int(cand) for cand in candidates]
            evidence_doc_ids = [int(ID) for ID in list(claim['evidence'].keys())]
            all_candidates = sorted(list(set(candidates + evidence_doc_ids)))
            if not train:
                missed_doc_ids = set(all_candidates).difference(set(candidates))
                all_candidates = candidates
                # Add missed_candidate to excluded_pairs?
                                
            for doc_id in all_candidates:
                doc = corpus[int(doc_id)]
                doc_id = str(doc_id)

                if "discourse" in doc:
                    abstract_sentences = \
                    [discourse + " " + sentence for discourse, sentence in zip(doc['discourse'], doc['abstract'])]
                else:
                    abstract_sentences = [sent.strip() for sent in doc['abstract']]
                
                if train:
                    if doc_id in claim['evidence']:
                        evidence = claim['evidence'][doc_id]
                        evidence_sentence_idx = {s for es in evidence for s in es['sentences']}
                        evidence_sentence_idx_sets = [set(es['sentences']) for es in evidence]
                        stances = set([es["label"] for es in evidence])
                        if "SUPPORT" in stances:
                            stance = "SUPPORT"
                        elif "CONTRADICT" in stances:
                            stance = "CONTRADICT"
                        else:
                            stance = "NEI"
                    else: 
                        evidence_sentence_idx = set([])
                        stance = "NEI"
                    
                    if len(evidence_sentence_idx) == 0:
                        concat_sentences = "@"
                        rationale_label_string = "0"
                        
                        self.samples.append({
                            'dataset': 1,
                            'claim': claim['claim'],
                            'claim_id': claim['id'],
                            'doc_id': doc['doc_id'],
                            'paragraph': concat_sentences,
                            'label': rationale_label_string,
                            'stance': self.stance_ind["NEI"]
                        })
                        
                    else:
                        # Full-evidence sentences
                        evidence_sentences = []
                        for i in range(len(abstract_sentences)):
                            if i in evidence_sentence_idx:
                                evidence_sentences.append(abstract_sentences[i])
                        concat_sentences = (" "+sep_token+" ").join(evidence_sentences)
                        concat_sentences = clean_num(clean_url(concat_sentences))
                        rationale_label_string = "1"*len(evidence_sentence_idx)

                        self.samples.append({
                            'dataset': 1,
                            'claim': claim['claim'],
                            'claim_id': claim['id'],
                            'doc_id': doc['doc_id'],
                            'paragraph': concat_sentences,
                            'label': rationale_label_string,
                            'stance': self.stance_ind[stance]
                        })
                        
                        # Each evidence sentence set
                        for es_idx in evidence_sentence_idx_sets:
                            evidence_sentences = []
                            for i in range(len(abstract_sentences)):
                                if i in es_idx:
                                    evidence_sentences.append(abstract_sentences[i])
                            concat_sentences = (" "+sep_token+" ").join(evidence_sentences)
                            concat_sentences = clean_num(clean_url(concat_sentences))
                            rationale_label_string = "1"*len(evidence_sentence_idx)

                            self.samples.append({
                                'dataset': 1,
                                'claim': claim['claim'],
                                'claim_id': claim['id'],
                                'doc_id': doc['doc_id'],
                                'paragraph': concat_sentences,
                                'label': rationale_label_string,
                                'stance': self.stance_ind[stance]
                            })
                            
                    # Negative sentences for both positive and negative paragraphs
                    non_rationale_idx = set(range(len(abstract_sentences))) - evidence_sentence_idx
                    non_rationale_idx = random.sample(non_rationale_idx,
                                                      k=min(random.randint(1, 3), len(non_rationale_idx)))
                    non_rationale_sentences = [abstract_sentences[i].strip() for i in sorted(list(non_rationale_idx))]
                    
                    concat_sentences = (" "+sep_token+" ").join(non_rationale_sentences)
                    concat_sentences = clean_num(clean_url(concat_sentences))
                    rationale_label_string = "0"*len(non_rationale_sentences)
                    
                    self.samples.append({
                        'dataset': 1,
                        'claim': claim['claim'],
                        'claim_id': claim['id'],
                        'doc_id': doc['doc_id'],
                        'paragraph': concat_sentences,
                        'label': rationale_label_string,
                        'stance': self.stance_ind["NEI"]
                    })
                    
                else:
                    if len(evidence_sentence_idx) == 0:
                        concat_sentences = "@"
                    else:
                        evidence_sentences = []
                        for i in range(len(abstract_sentences)):
                            if i in evidence_sentence_idx:
                                evidence_sentences.append(abstract_sentences[i])
                        concat_sentences = (" "+sep_token+" ").join(evidence_sentences)
                        concat_sentences = clean_num(clean_url(concat_sentences))

                    self.samples.append({
                        'dataset': 1,
                        'claim': claim['claim'],
                        'claim_id': claim['id'],
                        'doc_id': doc['doc_id'],
                        'paragraph': concat_sentences,
                    }) 

                        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
class FEVERStanceDataset(Dataset):
    """
    Dataset for a feeding a paragraph to a single BERT model.
    """
    def __init__(self, datapath: str, sep_token="</s>", train = True, k = 0):
        self.label_ind = {"NEI": 0, "rationale": 1}
        self.rev_label_ind = {i: l for (l, i) in self.label_ind.items()}
        self.stance_ind = {"NOT ENOUGH INFO": 0, "SUPPORTS": 1, "REFUTES": 2}
        self.rev_stance_ind = {i: l for (l, i) in self.stance_ind.items()}
        
        self.samples = []
        self.excluded_pairs = []
        
        def max_sent_len(sentences):
            return max([len(sent.strip().split()) for sent in sentences])
        
        for data in jsonlines.open(datapath):
            try:
                if len(data["sentences"]) > 0:
                    sentences = [sent.strip() for sent in data["sentences"]]
                    if max_sent_len(sentences) > 100 or len(sentences) > 100:
                        continue

                    if train:
                        rationales = []
                        rationale_sets = []
                        for evid in data["evidence_sets"]:
                            rationales.extend(evid)
                            rationale_sets.append(set(evid))
                        evidence_idx = set(rationales)
                        evidence_sentences = []
                        for i in range(len(sentences)):
                            if i in evidence_idx:
                                evidence_sentences.append(sentences[i])

                        # Full evidence sentencees
                        concat_sentences = (" "+sep_token+" ").join(evidence_sentences)
                        concat_sentences = clean_num(clean_url(concat_sentences))

                        self.samples.append({
                            'dataset': 0,
                            'claim': data['claim'],
                            'claim_id': data['id'],
                            'paragraph': concat_sentences,
                            'stance': self.stance_ind[data["label"]]
                        })

                        # For each evidence set
                        for evidence_set_idx in rationale_sets:
                            evidence_idx = set(evidence_set_idx)
                            evidence_sentences = []
                            for i in range(len(sentences)):
                                if i in evidence_idx:
                                    evidence_sentences.append(sentences[i])

                            concat_sentences = (" "+sep_token+" ").join(evidence_sentences)
                            concat_sentences = clean_num(clean_url(concat_sentences))

                            self.samples.append({
                                'dataset': 0,
                                'claim': data['claim'],
                                'claim_id': data['id'],
                                'paragraph': concat_sentences,
                                'stance': self.stance_ind[data["label"]]
                            })

                        # Negative sentences for both positive and negative paragraphs
                        non_rationale_idx = set(range(len(sentences))) - evidence_idx
                        non_rationale_idx = random.sample(non_rationale_idx,
                                                          k=min(random.randint(1, 3), len(non_rationale_idx)))
                        non_rationale_sentences = [sentences[i].strip() for i in sorted(list(non_rationale_idx))]

                        concat_sentences = (" "+sep_token+" ").join(non_rationale_sentences)
                        concat_sentences = clean_num(clean_url(concat_sentences))

                        self.samples.append({
                            'dataset': 1,
                            'claim': claim['claim'],
                            'claim_id': claim['id'],
                            'doc_id': doc['doc_id'],
                            'paragraph': concat_sentences,
                            'label': rationale_label_string,
                            'stance': self.stance_ind["NOT ENOUGH INFO"]
                        })

                    elif data["hit"]: # The retrieved pages hit the gold page.
                        concat_sentences = (" "+sep_token+" ").join(sentences)
                        concat_sentences = clean_num(clean_url(concat_sentences))
                        self.samples.append({
                            'dataset': 0,
                            'claim': data['claim'],
                            'claim_id': data['id'],
                            'paragraph': concat_sentences
                        })
            except:
                pass
               

                        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]