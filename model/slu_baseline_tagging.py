#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from transformers import AutoTokenizer, BertModel

class SLUTagging(nn.Module):

    def __init__(self, config):
        super(SLUTagging, self).__init__()
        self.config = config
        self.cell = config.encoder_cell # 保存用于RNN的单元类型（如LSTM或GRU）。
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)# 嵌入层，用于将单词ID转换成嵌入向量。嵌入大小为 config.embed_size，词汇表大小为 config.vocab_size
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout) # 丢弃层，用于减少过拟合。
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)# 一个用于标记的前馈神经网络（FNN）解码器，用于将RNN的输出转换为标签。

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths
        embed = self.word_embed(input_ids) # 将输入ID转换为嵌入向量
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True) # 为了高效处理不同长度的序列，输入被打包成一个 PackedSequence 对象。
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim.经过双向RNN处理，输出 packed_rnn_out（打包的RNN输出）和隐藏状态 h_t_c_t。
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True) # 使用 pad_packed_sequence 将RNN的输出解包回原始的序列格式。
        hiddens = self.dropout_layer(rnn_out) #[batchsize,序列长度,hidden size]
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids) # [batch size,序列长度,标签总数]
        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)
        prob = output[0]
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        if len(output) == 1:
            return predictions
        else:
            loss = output[1]
            return predictions, labels, loss.cpu().item()

class SLU_DE_Tagging(nn.Module): # dynamic embedding

    def __init__(self, config):
        super(SLU_DE_Tagging, self).__init__()
        self.config = config
        self.cell = config.encoder_cell # 保存用于RNN的单元类型（如LSTM或GRU）。        
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True) # 支持的类型包括LSTM，GRU，RNN
        self.dropout_layer = nn.Dropout(p=config.dropout) # 丢弃层，用于减少过拟合。
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)# 一个用于标记的前馈神经网络（FNN）解码器，用于将RNN的输出转换为标签。
        if config.pretrained_model == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('/mnt/workspace/Project/bert-base-chinese')
            self.bertmodel = BertModel.from_pretrained("/mnt/workspace/Project/bert-base-chinese")
        elif config.pretrained_model == 'bertw':
            self.tokenizer = AutoTokenizer.from_pretrained('/mnt/workspace/Project/chinese-bert-wwm-ext')
            self.bertmodel = BertModel.from_pretrained("/mnt/workspace/Project/chinese-bert-wwm-ext")
        elif config.pretrained_model == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('/mnt/workspace/Project/chinese-roberta-wwm-ext')
            self.bertmodel = BertModel.from_pretrained("/mnt/workspace/Project/chinese-roberta-wwm-ext")
        elif config.pretrained_model == 'macbert':
            self.tokenizer = AutoTokenizer.from_pretrained('/mnt/workspace/Project/chinese-macbert-base')
            self.bertmodel = BertModel.from_pretrained("/mnt/workspace/Project/chinese-macbert-base")
        self.device = config.device

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        utts = batch.bert_utts
        lengths = batch.lengths

        # BERT embedding
        encoded_input = self.tokenizer(utts, return_tensors='pt', padding=True) # 返回pytorch张量
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()} # 移动数据到指定设备
        embed = self.bertmodel(**encoded_input) # **：参数解包操作 # 维度含义：[batch size,seq length, hidden_size]
        embed = embed['last_hidden_state'][:,1:-1,:] #序列去掉首尾
        #rnn
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True) # 为了高效处理不同长度的序列，输入被打包成一个 PackedSequence 对象。
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim.经过双向RNN处理，输出 packed_rnn_out（打包的RNN输出）和隐藏状态 h_t_c_t。
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True) # 使用 pad_packed_sequence 将RNN的输出解包回原始的序列格式。
        hiddens = self.dropout_layer(rnn_out) #[batchsize,序列长度,hidden size]
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids) # [batch size,序列长度,标签总数]
        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)
        prob = output[0]
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        if len(output) == 1:
            return predictions
        else:
            loss = output[1]
            return predictions, labels, loss.cpu().item()

class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id): # config.hidden_size, config.num_tags, config.tag_pad_idx
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return (prob, )
