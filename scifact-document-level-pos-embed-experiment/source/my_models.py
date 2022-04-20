# -*- coding: utf-8 -*
# @Time    : 2021/8/14 17:28
# @Author  : gzy
# @File    : my_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup, AutoModelForSequenceClassification
from tqdm import tqdm
from util import read_passages, clean_words, test_f1, to_BIO, from_BIO

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class TimeDistributedDense(nn.Module):
    def __init__(self, INPUT_SIZE, OUTPUT_SIZE):
        super(TimeDistributedDense, self).__init__()
        self.input_size = INPUT_SIZE
        self.output_size = OUTPUT_SIZE
        self.linear = nn.Linear(INPUT_SIZE, OUTPUT_SIZE, bias=True)
        self.timedistributedlayer = TimeDistributed(self.linear)

    def forward(self, x):
        # x: (BATCH_SIZE, ARRAY_LEN, INPUT_SIZE)

        return self.timedistributedlayer(x)


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, num_labels, hidden_dropout_prob=0.1):
        super().__init__()
        self.dense = TimeDistributedDense(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = TimeDistributedDense(hidden_size, num_labels)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class WordAttention(nn.Module):
    """
    x: (BATCH_SIZE, N_sentence, N_token, INPUT_SIZE)
    token_mask: (batch_size, N_sep, N_token)
    out: (BATCH_SIZE, N_sentence, INPUT_SIZE)
    mask: (BATCH_SIZE, N_sentence)
    """
    def __init__(self, INPUT_SIZE, PROJ_SIZE, dropout=0.1):
        super(WordAttention, self).__init__()
        self.activation = torch.tanh
        self.att_proj = TimeDistributedDense(INPUT_SIZE, PROJ_SIZE)
        self.dropout = nn.Dropout(dropout)
        self.att_scorer = TimeDistributedDense(PROJ_SIZE, 1)

    def forward(self, x, token_mask):
        proj_input = self.att_proj(self.dropout(x.view(-1, x.size(-1))))
        proj_input = self.dropout(self.activation(proj_input))
        raw_att_scores = self.att_scorer(proj_input).squeeze(-1).view(x.size(0), x.size(1),
                                                                      x.size(2))  # (Batch_size, N_sentence, N_token)
        att_scores = F.softmax(raw_att_scores.masked_fill((1 - token_mask).bool(), float('-inf')), dim=-1)
        att_scores = torch.where(torch.isnan(att_scores), torch.zeros_like(att_scores),
                                 att_scores)  # Replace NaN with 0
        batch_att_scores = att_scores.view(-1, att_scores.size(-1))  # (Batch_size * N_sentence, N_token)
        out = torch.bmm(batch_att_scores.unsqueeze(1), x.view(-1, x.size(2), x.size(3))).squeeze(1)
        # (Batch_size * N_sentence, INPUT_SIZE)
        out = out.view(x.size(0), x.size(1), x.size(-1))
        mask = token_mask[:, :, 0]
        return out, mask


class DynamicSentenceAttention(nn.Module):
    """
    input: (BATCH_SIZE, N_sentence, INPUT_SIZE)
    output: (BATCH_SIZE, INPUT_SIZE)
    """

    def __init__(self, INPUT_SIZE, PROJ_SIZE, REC_HID_SIZE=None, dropout=0.1):
        super(DynamicSentenceAttention, self).__init__()
        self.activation = torch.tanh
        self.att_proj = TimeDistributedDense(INPUT_SIZE, PROJ_SIZE)
        self.dropout = nn.Dropout(dropout)

        if REC_HID_SIZE is not None:
            self.contextualized = True
            self.lstm = nn.LSTM(PROJ_SIZE, REC_HID_SIZE, bidirectional=False, batch_first=True)
            self.att_scorer = TimeDistributedDense(REC_HID_SIZE, 2)
        else:
            self.contextualized = False
            self.att_scorer = TimeDistributedDense(PROJ_SIZE, 2)

    def forward(self, sentence_reps, sentence_mask, att_scores, valid_scores):
        # sentence_reps: (BATCH_SIZE, N_sentence, INPUT_SIZE)
        # sentence_mask: (BATCH_SIZE, N_sentence)
        # att_scores: (BATCH_SIZE, N_sentence)
        # valid_scores: (BATCH_SIZE, N_sentence)
        # result: (BATCH_SIZE, INPUT_SIZE)
        # att_scores = rationale_out[:,:,1] # (BATCH_SIZE, N_sentence)
        # valid_scores = rationale_out[:,:,1] > rationale_out[:,:,0] # Only consider sentences predicted as rationales
        sentence_mask = torch.logical_and(sentence_mask, valid_scores)

        # Force those sentence representations in paragraph without rationale to be 0.
        # NEI_mask = (torch.sum(sentence_mask, axis=1) > 0).long().unsqueeze(-1).expand(-1, sentence_reps.size(-1))

        if sentence_reps.size(0) > 0:
            att_scores = F.softmax(att_scores.masked_fill((~sentence_mask).bool(), -1e4), dim=-1)
            # att_scores = torch.where(torch.isnan(att_scores), torch.zeros_like(att_scores), att_scores) # Replace NaN with 0
            result = torch.bmm(att_scores.unsqueeze(1), sentence_reps).squeeze(1)
            return result  # * NEI_mask
        else:
            return sentence_reps[:, 0, :]  # * NEI_mask


class ArgClassificationHead(nn.Module):
    """Head for sentence-level classification with arg feature tasks."""

    def __init__(self, hidden_size, arg_feature_size, sentence_repr_size, num_labels, hidden_dropout_prob = 0.1):
        super().__init__()
        #self.dense = TimeDistributedDense(hidden_size, 512)
        self.dense_proj = TimeDistributedDense(hidden_size, sentence_repr_size)
        self.arg_dense = TimeDistributedDense(arg_feature_size, arg_feature_size)
        #self.context_arg_dense = TimeDistributedDense(arg_feature_size*3, arg_feature_size*3)

        self.dropout = nn.Dropout(hidden_dropout_prob)
        #self.out_proj = TimeDistributedDense(sentence_repr_size+arg_feature_size, sentence_repr_size)
        self.out_proj = TimeDistributedDense(sentence_repr_size+arg_feature_size, num_labels)

        #self.output_dense = TimeDistributedDense(sentence_repr_size, num_labels)

    def forward(self, x, arg_feature, **kwargs):
        x = self.dropout(x)
        x = self.dense_proj(x)
        x = torch.tanh(x)
        x = self.dropout(x)

        # x = self.dense_proj(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)

        #arg_feature, context_arg_feature = arg_feature[0], arg_feature[1]
        #arg_feature = arg_feature

        arg = self.dropout(arg_feature)
        arg = self.arg_dense(arg)
        arg = torch.tanh(arg)
        arg = self.dropout(arg)

        # context_arg = self.dropout(context_arg_feature)
        # context_arg = self.context_arg_dense(context_arg)
        # context_arg = torch.tanh(context_arg)
        # context_arg = self.dropout(context_arg)

        concated_x = torch.cat([x, arg],2)
        # 这再加一层呢
        # out = self.out_proj(concated_x)
        # out = torch.tanh(out)
        # y = self.output_dense(out)

        y = self.out_proj(concated_x)
        return y


class ArgRationaleParagraphClassifier(nn.Module):
    def __init__(self, bert_path, bert_dim, dropout=0.1, arg_feature_size=5, sentence_repr_size=1024, ignore_index=2):
        super(ArgRationaleParagraphClassifier, self).__init__()
        self.rationale_label_size = 2
        self.ignore_index = 2
        self.bert = AutoModel.from_pretrained(bert_path)
        self.rationale_criterion = nn.CrossEntropyLoss(ignore_index=2)
        self.dropout = dropout
        self.bert_dim = bert_dim
        self.sentence_attention = DynamicSentenceAttention(bert_dim, bert_dim, dropout=dropout)
        self.word_attention = WordAttention(bert_dim, bert_dim, dropout=dropout)

        self.arg_feature_size = arg_feature_size
        self.sentence_repr_size = sentence_repr_size
        self.arg_linear = ArgClassificationHead(bert_dim, self.arg_feature_size, self.sentence_repr_size, self.rationale_label_size, hidden_dropout_prob=dropout)
        #self.rationale_linear = ClassificationHead(bert_dim, self.rationale_label_size, hidden_dropout_prob=dropout)

        self.extra_modules = [
            self.sentence_attention,
            self.word_attention,
            self.arg_linear,
            self.rationale_criterion
        ]

    def reinitialize(self):
        self.extra_modules = []
        self.arg_linear = ArgClassificationHead(self.bert_dim, self.rationale_label_size,
                                                   hidden_dropout_prob=self.dropout)
        self.sentence_attention = DynamicSentenceAttention(self.bert_dim, self.bert_dim, dropout=self.dropout)
        self.extra_modules = [
            self.rationale_linear,
            self.rationale_criterion,
            self.word_attention,
            self.sentence_attention
        ]

    def forward(self, encoded_dict, transformation_indices,  arg_feature, rationale_label=None, sample_p=1, rationale_score=False):
        batch_indices, indices_by_batch, mask = transformation_indices  # (batch_size, N_sep, N_token)
        bert_out = self.bert(**encoded_dict)[0]  # (BATCH_SIZE, sequence_len, BERT_DIM)
        bert_tokens = bert_out[batch_indices, indices_by_batch, :]
        #print('bert_tokens', bert_tokens.shape, bert_tokens)
        #print(mask.shape, mask)
        # bert_tokens: (batch_size, N_sep, N_token, BERT_dim)

        # att 机制改一下， 用rnn
        sentence_reps, sentence_mask = self.word_attention(bert_tokens, mask)
        # (Batch_size, N_sep, BERT_DIM), (Batch_size, N_sep)
        # print(bert_out.shape, bert_tokens.shape, sentence_reps.shape, sentence_mask.shape, rationale_label.shape)
        rationale_out = self.arg_linear(sentence_reps, arg_feature)  # (Batch_size, N_sep, 2)

        att_scores = rationale_out[:, :, 1]  # (BATCH_SIZE, N_sentence)

        if bool(torch.rand(1) < sample_p):  # Choose sentence according to predicted rationale
            valid_scores = rationale_out[:, :, 1] > rationale_out[:, :, 0]
        else:
            valid_scores = rationale_label == 1  # Ground truth
        paragraph_rep = self.sentence_attention(sentence_reps, sentence_mask, att_scores, valid_scores)
        # (BATCH_SIZE, BERT_DIM)

        if rationale_label is not None:
            rationale_loss = self.rationale_criterion(rationale_out.view(-1, self.rationale_label_size),
                                                      rationale_label.view(-1))  # ignore index 2
        else:
            rationale_loss = None

        if rationale_score:
            rationale_pred = rationale_out.cpu()[:, :, 1]  # (Batch_size, N_sep)
        else:
            rationale_pred = torch.argmax(rationale_out.cpu(), dim=-1)  # (Batch_size, N_sep)
        rationale_out = [rationale_pred_paragraph[mask].detach().numpy().tolist() for rationale_pred_paragraph, mask in
                         zip(rationale_pred, sentence_mask.bool())]
        return rationale_out, rationale_loss


class RationaleParagraphClassifier(nn.Module):
    def __init__(self, bert_path, bert_dim, dropout=0.1, ignore_index=2):
        super(RationaleParagraphClassifier, self).__init__()
        self.rationale_label_size = 2
        self.ignore_index = 2
        self.bert = AutoModel.from_pretrained(bert_path)
        self.rationale_criterion = nn.CrossEntropyLoss(ignore_index=2)
        self.dropout = dropout
        self.bert_dim = bert_dim
        self.sentence_attention = DynamicSentenceAttention(bert_dim, bert_dim, dropout=dropout)
        self.word_attention = WordAttention(bert_dim, bert_dim, dropout=dropout)
        self.rationale_linear = ClassificationHead(bert_dim, self.rationale_label_size, hidden_dropout_prob=dropout)
        self.extra_modules = [
            self.sentence_attention,
            self.word_attention,
            self.rationale_linear,
            self.rationale_criterion
        ]

    def reinitialize(self):
        self.extra_modules = []
        self.rationale_linear = ClassificationHead(self.bert_dim, self.rationale_label_size,
                                                   hidden_dropout_prob=self.dropout)
        self.sentence_attention = DynamicSentenceAttention(self.bert_dim, self.bert_dim, dropout=self.dropout)
        self.extra_modules = [
            self.rationale_linear,
            self.rationale_criterion,
            self.word_attention,
            self.sentence_attention
        ]

    def forward(self, encoded_dict, transformation_indices, rationale_label=None, sample_p=1, rationale_score=False):
        batch_indices, indices_by_batch, mask = transformation_indices  # (batch_size, N_sep, N_token)
        bert_out = self.bert(**encoded_dict)[0]  # (BATCH_SIZE, sequence_len, BERT_DIM)
        bert_tokens = bert_out[batch_indices, indices_by_batch, :]
        # bert_tokens: (batch_size, N_sep, N_token, BERT_dim)
        sentence_reps, sentence_mask = self.word_attention(bert_tokens, mask)
        # (Batch_size, N_sep, BERT_DIM), (Batch_size, N_sep)
        # print(bert_out.shape, bert_tokens.shape, sentence_reps.shape, sentence_mask.shape, rationale_label.shape)
        rationale_out = self.rationale_linear(sentence_reps)  # (Batch_size, N_sep, 2)
        #att_scores = rationale_out[:, :, 1]  # (BATCH_SIZE, N_sentence)

        if bool(torch.rand(1) < sample_p):  # Choose sentence according to predicted rationale
            valid_scores = rationale_out[:, :, 1] > rationale_out[:, :, 0]
        else:
            valid_scores = rationale_label == 1  # Ground truth
        #paragraph_rep = self.sentence_attention(sentence_reps, sentence_mask, att_scores, valid_scores)
        # (BATCH_SIZE, BERT_DIM)

        if rationale_label is not None:
            rationale_loss = self.rationale_criterion(rationale_out.view(-1, self.rationale_label_size),
                                                      rationale_label.view(-1))  # ignore index 2
        else:
            rationale_loss = None

        if rationale_score:
            rationale_pred = rationale_out.cpu()[:, :, 1]  # (Batch_size, N_sep)
        else:
            rationale_pred = torch.argmax(rationale_out.cpu(), dim=-1)  # (Batch_size, N_sep)
        rationale_out = [rationale_pred_paragraph[mask].detach().numpy().tolist() for rationale_pred_paragraph, mask in
                         zip(rationale_pred, sentence_mask.bool())]
        return rationale_out, rationale_loss


class DocumentPositionEncoder(nn.Module):
    def __init__(self,max_doc_length, pos_emd_dim):
        super(DocumentPositionEncoder, self).__init__()
        self.pos_emd_dim = pos_emd_dim
        self.max_doc_length = max_doc_length
        self.position_embedding = nn.Parameter(torch.randn(max_doc_length, pos_emd_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.position_embedding)

    def forward(self, x, mask):
        max_length=x.size(1)
        batch = x.size(0)
        pos_embed = self.position_embedding[:max_length].view(1, max_length, self.pos_emd_dim)

        # pos_embed = self.position_embedding[:max_length].expand(batch, max_length, self.pos_emd_dim)
        # mask = mask.repeat(max_length, self.pos_emd_dim)
        # pos_embed = torch.bmm(pos_embed, mask)
        return x + pos_embed


class ParagraphPosEmbRationaleParagraphClassifier(nn.Module):
    def __init__(self, bert_path, bert_dim, dropout, max_doc_length, pos_emd_dim):
        super(ParagraphPosEmbRationaleParagraphClassifier, self).__init__()
        self.rationale_label_size = 2
        self.ignore_index = 2
        self.bert = AutoModel.from_pretrained(bert_path)
        self.rationale_criterion = nn.CrossEntropyLoss(ignore_index=2)
        self.dropout = dropout
        self.bert_dim = bert_dim
        self.sentence_attention = DynamicSentenceAttention(bert_dim, bert_dim, dropout=dropout)
        self.word_attention = WordAttention(bert_dim, bert_dim, dropout=dropout)
        self.rationale_linear = ClassificationHead(bert_dim, self.rationale_label_size, hidden_dropout_prob=dropout)
        self.position_encoder = DocumentPositionEncoder(max_doc_length, pos_emd_dim)
        #self.position_embedding = nn.Parameter(torch.randn(max_doc_length, pos_emd_dim))
        self.extra_modules = [
            self.sentence_attention,
            self.word_attention,
            self.rationale_linear,
            self.rationale_criterion
        ]

    def reinitialize(self):
        self.extra_modules = []
        self.rationale_linear = ClassificationHead(self.bert_dim, self.rationale_label_size,
                                                   hidden_dropout_prob=self.dropout)
        self.sentence_attention = DynamicSentenceAttention(self.bert_dim, self.bert_dim, dropout=self.dropout)
        self.extra_modules = [
            self.rationale_linear,
            self.rationale_criterion,
            self.word_attention,
            self.sentence_attention
        ]

    def forward(self, encoded_dict, transformation_indices, pos_feature,rationale_label=None, sample_p=1, rationale_score=False):
        batch_indices, indices_by_batch, mask = transformation_indices  # (batch_size, N_sep, N_token)
        bert_out = self.bert(**encoded_dict)[0]  # (BATCH_SIZE, sequence_len, BERT_DIM)
        bert_tokens = bert_out[batch_indices, indices_by_batch, :]
        # bert_tokens: (batch_size, N_sep, N_token, BERT_dim)
        sentence_reps, sentence_mask = self.word_attention(bert_tokens, mask)
        # (Batch_size, N_sep, BERT_DIM), (Batch_size, N_sep)

        sentence_reps = self.position_encoder(sentence_reps, sentence_mask)

        rationale_out = self.rationale_linear(sentence_reps)  # (Batch_size, N_sep, 2)
        #att_scores = rationale_out[:, :, 1]  # (BATCH_SIZE, N_sentence)

        if bool(torch.rand(1) < sample_p):  # Choose sentence according to predicted rationale
            valid_scores = rationale_out[:, :, 1] > rationale_out[:, :, 0]
        else:
            valid_scores = rationale_label == 1  # Ground truth
        #paragraph_rep = self.sentence_attention(sentence_reps, sentence_mask, att_scores, valid_scores)
        # (BATCH_SIZE, BERT_DIM)

        if rationale_label is not None:
            rationale_loss = self.rationale_criterion(rationale_out.view(-1, self.rationale_label_size),
                                                      rationale_label.view(-1))  # ignore index 2
        else:
            rationale_loss = None

        if rationale_score:
            rationale_pred = rationale_out.cpu()[:, :, 1]  # (Batch_size, N_sep)
        else:
            rationale_pred = torch.argmax(rationale_out.cpu(), dim=-1)  # (Batch_size, N_sep)
        rationale_out = [rationale_pred_paragraph[mask].detach().numpy().tolist() for rationale_pred_paragraph, mask in
                         zip(rationale_pred, sentence_mask.bool())]
        return rationale_out, rationale_loss