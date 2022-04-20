# -*- coding: utf-8 -*
# @Time    : 2021/8/9 11:02
# @Author  : gzy
# @File    : my_models.py
import torch
from torch import nn
from  torch.nn import CrossEntropyLoss, MSELoss
from transformers import AutoModel,AutoTokenizer,AutoConfig


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MyBertForSequenceClassification(nn.Module):

    def __init__(self, rep_file, args):
        super(MyBertForSequenceClassification, self).__init__()
        self.bert = AutoModel.from_pretrained(rep_file)
        self.num_labels = args.num_labels

        self.dropout = nn.Dropout(args.lstm_dropout)
        # self.pooling = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(args.lstm_hidden_size * 2, args.num_labels)
        if args.freeze is not None:
            self.freeze = args.freeze
        else:
            self.freeze = 0

        print("--------- this is freeze: " + str(self.freeze) + " ----------")
        self.W = []
        self.gru = []
        for i in range(args.lstm_layers):
            self.W.append(nn.Linear(args.lstm_hidden_size * 2, args.lstm_hidden_size * 2))
            self.gru.append(
                nn.GRU(args.bert_hidden_size, args.lstm_hidden_size,
                       num_layers=1, bidirectional=True, batch_first=True).cuda())
        self.W = nn.ModuleList(self.W)
        self.gru = nn.ModuleList(self.gru)
        self.apply(self.init_weights)

        self.extra_modules = [
            self.classifier,
            self.W,
            self.gru
        ]

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        if self.freeze == 1:
            with torch.no_grad():
                outputs = self.bert(input_ids=flat_input_ids, position_ids=flat_position_ids,
                                    token_type_ids=flat_token_type_ids,
                                    attention_mask=flat_attention_mask, head_mask=head_mask)
        else:
            outputs = self.bert(input_ids=flat_input_ids, position_ids=flat_position_ids,
                                attention_mask=flat_attention_mask)
        pooled_output = outputs[1]

        output = pooled_output.reshape(input_ids.size(0), input_ids.size(1), -1).contiguous()

        for w, gru in zip(self.W, self.gru):
            gru.flatten_parameters()
            output, hidden = gru(output)
            output = self.dropout(output)

        hidden = hidden.permute(1, 0, 2).reshape(input_ids.size(0), -1).contiguous()
        # hidden=output.mean(1)
        # hidden=nn.functional.tanh(self.pooling(hidden))
        # hidden=self.dropout(hidden)
        logits = self.classifier(hidden)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss, ]
            outputs = outputs + [nn.functional.softmax(logits, -1)]
        else:
            outputs = nn.functional.softmax(logits, -1)

        return outputs  # (loss), logits, (hidden_states), (attentions)


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.bert_hidden_size, args.bert_hidden_size)
        self.dropout = nn.Dropout(0.0)
        self.out_proj = nn.Linear(args.bert_hidden_size, args.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class ArgFeatureRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks with arg features"""

    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.bert_hidden_size, args.bert_hidden_size)
        self.arg_dense = nn.Linear(6, 6)
        self.dropout = nn.Dropout(0.0)
        self.out_proj = nn.Linear(args.bert_hidden_size+6, args.num_labels)

    def forward(self, features, arg_feature, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        #x = self.out_proj(x)

        arg = self.dropout(arg_feature)
        arg = self.arg_dense(arg)
        arg = torch.tanh(arg)
        arg = self.dropout(arg)

        concat_features = torch.cat([x, arg], dim=1)
        y = self.out_proj(concat_features)

        return y

class ArgBertForSequenceClassification(nn.Module):

    def __init__(self, rep_file, args):
        super(ArgBertForSequenceClassification, self).__init__()
        self.bert = AutoModel.from_pretrained(rep_file)
        self.num_labels = args.num_labels
        self.classifier = ArgFeatureRobertaClassificationHead(args)
        self.extra_modules = [
            self.classifier
        ]

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, arg_feature=None,
                position_ids=None, head_mask=None):
        #print(input_ids,attention_mask,labels)

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        outputs = self.bert(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        pooled_output = outputs[0]
        #output = pooled_output.reshape(input_ids.size(0), input_ids.size(1), -1).contiguous()

        logits = self.classifier(pooled_output, arg_feature)

        #print('预测输出',np.argmax(logits.detach().cpu().numpy(), axis=1).tolist())

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss, ]
            outputs = outputs + [nn.functional.softmax(logits, -1)]
        else:
            outputs = nn.functional.softmax(logits, -1)

        return outputs  # (loss), logits, (hidden_states), (attentions)

class TestBertForSequenceClassification(nn.Module):

    def __init__(self, rep_file, args):
        super(TestBertForSequenceClassification, self).__init__()
        self.bert = AutoModel.from_pretrained(rep_file)
        self.num_labels = args.num_labels

        #self.dropout = nn.Dropout(args.lstm_dropout)
        # self.pooling = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = RobertaClassificationHead(args)


        self.extra_modules = [
            self.classifier
            # self.W,
            # self.gru
        ]

    # 初始化权重函数 是配合from_pretrained()函数使用的, 他会清空加载的bert模型的参数
    # 为什么准确率一直是23.1，因为我的代码是再model构造函数里，先Load bert参数，然后又调用init_weights
    # 这样bert参数就清空了，怪不得预测的label都是2了。
    # RobertaForSequenceClassification.from_pretrained('roberta-base', return_dict=True)
    # 如果像我这样不用from pretrained

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        #print(input_ids,attention_mask,labels)

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        outputs = self.bert(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        pooled_output = outputs[0]

        #output = pooled_output.reshape(input_ids.size(0), input_ids.size(1), -1).contiguous()

        logits = self.classifier(pooled_output)

        import numpy as np

        #print('预测输出',np.argmax(logits.detach().cpu().numpy(), axis=1).tolist())

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss, ]
            outputs = outputs + [nn.functional.softmax(logits, -1)]
        else:
            outputs = nn.functional.softmax(logits, -1)

        return outputs  # (loss), logits, (hidden_states), (attentions)

class NaiveClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.lstm_hidden_size*2, args.lstm_hidden_size*2)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(args.lstm_hidden_size*2, args.num_labels)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class TestBertGRUForSequenceClassification(nn.Module):

    def __init__(self, rep_file, args):
        super(TestBertGRUForSequenceClassification, self).__init__()
        self.bert = AutoModel.from_pretrained(rep_file)
        self.num_labels = args.num_labels
        self.dropout = nn.Dropout(args.lstm_dropout)
        self.classifier = NaiveClassificationHead(args)

        self.W = []
        self.gru = []
        for i in range(args.lstm_layers):
            self.W.append(nn.Linear(args.lstm_hidden_size * 2, args.lstm_hidden_size * 2))
            self.gru.append(
                nn.GRU(args.bert_hidden_size, args.lstm_hidden_size,
                       num_layers=1, bidirectional=True, batch_first=True).cuda())
        self.W = nn.ModuleList(self.W)
        self.gru = nn.ModuleList(self.gru)

        self.extra_modules = [
            self.classifier,
            self.W,
            self.gru
        ]

    # 初始化权重函数 是配合from_pretrained()函数使用的, 他会清空加载的bert模型的参数
    # 为什么准确率一直是23.1，因为我的代码是再model构造函数里，先Load bert参数，然后又调用init_weights
    # 这样bert参数就清空了，怪不得预测的label都是2了。
    # RobertaForSequenceClassification.from_pretrained('roberta-base', return_dict=True)
    # 如果像我这样不用from pretrained

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        #print(input_ids,attention_mask,labels)

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        outputs = self.bert(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        pooled_output = outputs[1]

        output = pooled_output.reshape(input_ids.size(0), input_ids.size(1), -1).contiguous()

        for w, gru in zip(self.W, self.gru):
            gru.flatten_parameters()
            output, hidden = gru(output)
            output = self.dropout(output)

        hidden = hidden.permute(1, 0, 2).reshape(input_ids.size(0), -1).contiguous()

        # hidden=output.mean(1)
        # hidden=nn.functional.tanh(self.pooling(hidden))
        # hidden=self.dropout(hidden)

        logits = self.classifier(hidden)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss, ]
            outputs = outputs + [nn.functional.softmax(logits, -1)]
        else:
            outputs = nn.functional.softmax(logits, -1)

        return outputs  # (loss), logits, (hidden_states), (attentions)

class TestBertGRUForSequenceClassification2(nn.Module):

    def __init__(self, rep_file, args):
        super(TestBertGRUForSequenceClassification2, self).__init__()
        self.bert = AutoModel.from_pretrained(rep_file)
        self.num_labels = args.num_labels
        self.dropout = nn.Dropout(args.lstm_dropout)
        #self.classifier = NaiveClassificationHead(args)
        self.classifier = nn.Linear(args.lstm_hidden_size * 2,args.num_labels)

        self.W = []
        self.gru = []
        for i in range(args.lstm_layers):
            self.W.append(nn.Linear(args.lstm_hidden_size * 2, args.lstm_hidden_size * 2))
            self.gru.append(
                nn.GRU(args.bert_hidden_size, args.lstm_hidden_size,
                       num_layers=1, bidirectional=True, batch_first=True).cuda())
        self.W = nn.ModuleList(self.W)
        self.gru = nn.ModuleList(self.gru)

        self.extra_modules = [
            self.classifier,
            self.W,
            self.gru
        ]

    # 初始化权重函数 是配合from_pretrained()函数使用的, 他会清空加载的bert模型的参数
    # 为什么准确率一直是23.1，因为我的代码是再model构造函数里，先Load bert参数，然后又调用init_weights
    # 这样bert参数就清空了，怪不得预测的label都是2了。
    # RobertaForSequenceClassification.from_pretrained('roberta-base', return_dict=True)
    # 如果像我这样不用from pretrained

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        #print(input_ids,attention_mask,labels)

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        outputs = self.bert(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        pooled_output = outputs[1]

        output = pooled_output.reshape(input_ids.size(0), input_ids.size(1), -1).contiguous()

        for w, gru in zip(self.W, self.gru):
            gru.flatten_parameters()
            output, hidden = gru(output)
            output = self.dropout(output)

        hidden = hidden.permute(1, 0, 2).reshape(input_ids.size(0), -1).contiguous()

        # hidden=output.mean(1)
        # hidden=nn.functional.tanh(self.pooling(hidden))
        # hidden=self.dropout(hidden)

        logits = self.classifier(hidden)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss, ]
            outputs = outputs + [nn.functional.softmax(logits, -1)]
        else:
            outputs = nn.functional.softmax(logits, -1)

        return outputs  # (loss), logits, (hidden_states), (attentions)

class Last2ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.bert_hidden_size*3, args.bert_hidden_size*3)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(args.bert_hidden_size*3, args.num_labels)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class TestBertLast2EmdForSequenceClassification(nn.Module):

    def __init__(self, rep_file, args):
        super(TestBertLast2EmdForSequenceClassification, self).__init__()
        self.bert = AutoModel.from_pretrained(rep_file)
        self.num_labels = args.num_labels
        self.dropout = nn.Dropout(args.lstm_dropout)
        self.classifier = Last2ClassificationHead(args)

        self.extra_modules = [
            self.classifier,
            # self.W,
            # self.gru
        ]

    # 初始化权重函数 是配合from_pretrained()函数使用的, 他会清空加载的bert模型的参数
    # 为什么准确率一直是23.1，因为我的代码是再model构造函数里，先Load bert参数，然后又调用init_weights
    # 这样bert参数就清空了，怪不得预测的label都是2了。
    # RobertaForSequenceClassification.from_pretrained('roberta-base', return_dict=True)
    # 如果像我这样不用from pretrained

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        #print(input_ids,attention_mask,labels)

        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None

        outputs = self.bert(input_ids=flat_input_ids, attention_mask=flat_attention_mask,output_hidden_states=True)

        pooled_output = outputs[1]
        hidden_states = outputs[-1]
        output = torch.cat([pooled_output, hidden_states[-1][:,0,:], hidden_states[-2][:,0,:]], dim=1)

        # output = pooled_output.reshape(input_ids.size(0), input_ids.size(1), -1).contiguous()
        #
        # for w, gru in zip(self.W, self.gru):
        #     gru.flatten_parameters()
        #     output, hidden = gru(output)
        #     output = self.dropout(output)
        #
        # hidden = hidden.permute(1, 0, 2).reshape(input_ids.size(0), -1).contiguous()

        # hidden=output.mean(1)
        # hidden=nn.functional.tanh(self.pooling(hidden))
        # hidden=self.dropout(hidden)

        logits = self.classifier(output)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss, ]
            outputs = outputs + [nn.functional.softmax(logits, -1)]
        else:
            outputs = nn.functional.softmax(logits, -1)

        return outputs  # (loss), logits, (hidden_states), (attentions)


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