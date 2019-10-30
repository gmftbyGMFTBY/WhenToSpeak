#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.9.14


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import random
import numpy as np
import ipdb

from .layers import *


'''
HRED model with the attention on context encoder

have the classification
'''

class Utterance_encoder_cf(nn.Module):

    '''
    Bidirectional GRU
    '''

    def __init__(self, input_size, embedding_size, 
                 hidden_size, dropout=0.5, n_layer=1, pretrained=None):
        super(Utterance_encoder_cf, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.n_layer = n_layer

        if pretrained:
            pretrained = f'{pretrained}/ipt_bert_embedding.pkl'
            self.embed = PretrainedEmbedding(input_size, embedding_size, pretrained)
        else:
            self.embed = nn.Embedding(input_size, self.embedding_size)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, num_layers=n_layer, 
                          dropout=dropout, bidirectional=True)
        # hidden_project
        self.hidden_proj = nn.Linear(n_layer * 2 * self.hidden_size, hidden_size)
        self.bn = nn.BatchNorm1d(num_features=hidden_size)

        self.init_weight()

    def init_weight(self):
        init.xavier_normal_(self.hidden_proj.weight)
        init.orthogonal_(self.gru.weight_hh_l0)
        init.orthogonal_(self.gru.weight_ih_l0)
        self.gru.bias_ih_l0.data.fill_(0.0)
        self.gru.bias_hh_l0.data.fill_(0.0)

    def forward(self, inpt, lengths, hidden=None):
        # use pack_padded
        # inpt: [seq_len, batch], lengths: [batch_size]
        embedded = self.embed(inpt)    # [seq_len, batch, input_size]

        if not hidden:
            hidden = torch.randn(self.n_layer * 2, len(lengths), 
                                 self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden.cuda()

        embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, enforce_sorted=False)
        _, hidden = self.gru(embedded, hidden)    
        # [n_layer * bidirection, batch, hidden_size]
        # hidden = hidden.reshape(hidden.shape[1], -1)
        # ipdb.set_trace()
        hidden = hidden.permute(1, 0, 2)    # [batch, n_layer * bidirectional, hidden_size]
        hidden = hidden.reshape(hidden.size(0), -1) # [batch, *]
        hidden = self.bn(self.hidden_proj(hidden))
        hidden = torch.tanh(hidden)   # [batch, hidden]
        return hidden


class Context_encoder_cf(nn.Module):

    '''
    input_size is 2 * utterance_hidden_size
    '''

    def __init__(self, input_size, hidden_size, dropout=0.5, user_embed_size=10):
        super(Context_encoder_cf, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.user_embed_size = user_embed_size
        self.gru = nn.GRU(self.input_size+user_embed_size, self.hidden_size)
        self.drop = nn.Dropout(p=dropout)

        self.init_weight()

    def init_weight(self):
        init.orthogonal_(self.gru.weight_hh_l0)
        init.orthogonal_(self.gru.weight_ih_l0)

    def forward(self, inpt, user_embed, hidden=None):
        # inpt: [turn_len, batch, input_size], user_embed: [turn_len, batch, user_embed]
        # hidden
        if not hidden:
            hidden = torch.randn(1, inpt.shape[1], self.hidden_size)
            if torch.cuda.is_available():
                hidden = hidden.cuda()

        # cat
        inpt = torch.cat([inpt, user_embed], 2)    # [turn_len, batch, inpt_size + user_embed]
        
        inpt = self.drop(inpt)
        output, hidden = self.gru(inpt, hidden)

        # hidden: [1, batch, hidden_size]
        hidden = hidden.squeeze(0)    # [batch, hidden_size]
        
        return output, hidden
        


class Decoder_cf(nn.Module):

    '''
    Max likelyhood for decoding the utterance
    input_size is the size of the input vocabulary

    Attention module should satisfy that the decoder_hidden size is the same as 
    the Context encoder hidden size
    '''

    def __init__(self, output_size, embed_size, hidden_size, pretrained=None):
        super(Decoder_cf, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        if pretrained:
            pretrained = f'{pretrained}/opt_bert_embedding.pkl'
            self.embed = PretrainedEmbedding(self.output_size, self.embed_size, pretrained)
        else:
            self.embed = nn.Embedding(self.output_size, self.embed_size)
        self.gru = nn.GRU(self.embed_size + self.hidden_size, 
                          self.hidden_size)

        self.out = nn.Linear(2 * hidden_size, output_size)
        # self.out = nn.Linear(2 * hidden_size, output_size)

        # attention on context encoder
        self.attn = Attention(hidden_size)

        self.init_weight()

    def init_weight(self):
        init.orthogonal_(self.gru.weight_hh_l0)
        init.orthogonal_(self.gru.weight_ih_l0)

    def forward(self, inpt, last_hidden, encoder_outputs):
        # inpt: [batch_size], last_hidden: [1, batch, hidden_size]
        # encoder_outputs: [turn_len, batch, hidden_size], user_de: [batch, 11]
        embedded = self.embed(inpt).unsqueeze(0)    # [1, batch_size, embed_size]
        last_hidden = last_hidden.squeeze(0)    # [batch, hidden]

        # [batch, 1, seq_len]
        attn_weights = self.attn(last_hidden, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1)    # [1, batch, hidden]

        rnn_input = torch.cat([embedded, context], 2)   # [1, batch, 2 * hidden]

        # output: [1, batch, hidden_size], hidden: [1, batch, hidden_size]
        output, hidden = self.gru(rnn_input, last_hidden.unsqueeze(0))
        output = output.squeeze(0)    # [batch, hidden_size]
        context = context.squeeze(0)  # [batch, hidden]
        output = torch.cat([output, context], 1)    # [batch, 2 * hidden]
        output = self.out(output)     # [batch, output_size]
        output = F.log_softmax(output, dim=1)
        return output, hidden


class HRED_cf(nn.Module):

    def __init__(self, embed_size, input_size, output_size, 
                 utter_hidden, context_hidden, decoder_hidden, 
                 teach_force=0.5, pad=24745, sos=24742, dropout=0.5, utter_n_layer=1,
                 pretrained=None, user_embed_size=10):
        super(HRED_cf, self).__init__()
        self.teach_force = teach_force
        self.output_size = output_size
        self.pad, self.sos = pad, sos
        self.utter_encoder = Utterance_encoder_cf(input_size, embed_size, utter_hidden, 
                                                  dropout=dropout, n_layer=utter_n_layer,
                                                  pretrained=pretrained)
        self.context_encoder = Context_encoder_cf(utter_hidden, context_hidden, 
                                                  dropout=dropout, 
                                                  user_embed_size=user_embed_size) 
        self.decoder = Decoder_cf(output_size, embed_size, decoder_hidden, 
                                  pretrained=pretrained)

        # user embedding, 10 embedding size
        self.user_emb = nn.Embedding(2, 10)

        # decision, binary classification
        self.decision_1 = nn.Linear(utter_hidden + user_embed_size, int(utter_hidden / 2))
        self.decision_2 = nn.Linear(int(utter_hidden / 2), 1)
        self.decision_drop = nn.Dropout(p=dropout)

        # hidden project
        self.hidden_proj = nn.Linear(decoder_hidden + user_embed_size, decoder_hidden)
        self.hidden_drop = nn.Dropout(p=dropout)

    def forward(self, src, tgt, subatch, tubatch, lengths):
        # src: [turns, lengths, batch], tgt: [lengths, batch]
        # subatch: [turns, batch], tubatch: [batch], lengths: [turns, batch]
        turn_size, batch_size, maxlen = len(src), tgt.size(1), tgt.size(0)
        outputs = torch.zeros(maxlen, batch_size, self.output_size)
        if torch.cuda.is_available():
            outputs = outputs.cuda()

        # user embedding
        subatch = self.user_emb(subatch)    # [turn_len, batch, 10]
        tubatch = self.user_emb(tubatch)    # [batch, 10]

        # utterance encoding
        turns = []
        for i in range(turn_size):
            # sbatch = src[i].transpose(0, 1)    # [seq_len, batch]
            hidden = self.utter_encoder(src[i], lengths[i])    # utter_hidden
            turns.append(hidden)
        turns = torch.stack(turns)    # [turn_len, batch, utter_hidden]

        # context encoding
        # input: [turn_len ,batch, utter_hidden], [turn_len, batch, embed_size]
        # output: [seq, batch, hidden], [batch, hidden]
        context_output, hidden = self.context_encoder(turns, subatch)

        # decision
        decision_inpt = torch.cat([hidden, tubatch], 1)    # [batch, hidden+10]
        de = self.decision_drop(torch.tanh(self.decision_1(decision_inpt)))
        de = torch.sigmoid(self.decision_2(de)).squeeze(1)    # [batch]

        # ========== decoding using he tgt_user & decision information ========== #
        # user_de = torch.cat([tubatch, de.unsqueeze(1)], 1)    # [batch, 11]

        # decoding
        # tgt = tgt.transpose(0, 1)        # [seq_len, batch]
        # ========== combine the hidden and the tbatch
        hidden = hidden.unsqueeze(0)     # [1, batch, hidden_size]
        hidden = torch.cat([hidden, tubatch.unsqueeze(0)], 2)     # [1, batch, hidden + 10 + 1]
        hidden = self.hidden_drop(torch.tanh(self.hidden_proj(hidden)))  # [1, batch, hidden]

        output = tgt[0, :]          # [batch]

        for i in range(1, maxlen):
            # output, hidden = self.decoder(output, hidden, context_output, user_de)
            output, hidden = self.decoder(output, hidden, context_output)
            outputs[i] = output
            is_teacher = random.random() < self.teach_force
            top1 = output.data.max(1)[1]
            if is_teacher:
                output = tgt[i].clone().detach()
            else:
                output = top1

        # de: [batch], outputs: [maxlen, batch, vocab_size]
        return de, outputs


    def predict(self, src, subatch, tubatch, maxlen, lengths):
        # predict for test dataset, return outputs: [maxlen, batch_size]
        # src: [turn, max_len, batch_size], lengths: [turn, batch_size]
        # subatch: [turn_len, batch], tubatch: [batch]
        turn_size, batch_size = len(src), src[0].size(1)
        outputs = torch.zeros(maxlen, batch_size)
        if torch.cuda.is_available():
            outputs = outputs.cuda()
        
        # user embedding
        subatch = self.user_emb(subatch)    # [turn_len, batch, 10]
        tubatch = self.user_emb(tubatch)    # [batch, 10]

        turns = []
        for i in range(turn_size):
            # sbatch = src[i].transpose(0, 1)
            hidden = self.utter_encoder(src[i], lengths[i])
            turns.append(hidden)
        turns = torch.stack(turns)
        
        # context encoding
        # input: [turn_len ,batch, utter_hidden], [turn_len, batch, embed_size]
        # output: [seq, batch, hidden], [batch, hidden]
        context_output, hidden = self.context_encoder(turns, subatch)
        
        # decision
        decision_inpt = torch.cat([hidden, tubatch], 1)    # [batch, hidden+10]
        de = self.decision_drop(torch.tanh(self.decision_1(decision_inpt)))
        de = torch.sigmoid(self.decision_2(de)).squeeze(1)    # [batch]

        # ========== decoding using he tgt_user & decision information ========== #
        # user_de = torch.cat([tubatch, de.unsqueeze(1)], 1)    # [batch, 11]

        hidden = hidden.unsqueeze(0)    # [1, batch, hidden]
        hidden = torch.cat([hidden, tubatch.unsqueeze(0)], 2)     # [1, batch, hidden + 10 + 1]
        hidden = self.hidden_drop(torch.tanh(self.hidden_proj(hidden)))  # [1, batch, hidden]

        output = torch.zeros(batch_size, dtype=torch.long).fill_(self.sos)
        if torch.cuda.is_available():
            output = output.cuda()

        for i in range(1, maxlen):
            # output, hidden = self.decoder(output, hidden, context_output, user_de)
            output, hidden = self.decoder(output, hidden, context_output)
            output = output.max(1)[1]
            outputs[i] = output

        # de: [batch], outputs: [maxlen, batch]
        return de, outputs


if __name__ == "__main__":
    pass
