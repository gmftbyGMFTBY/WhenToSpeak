#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.2.24

'''
Attention Layer
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.data import Data, DataLoader
import math
import random
import numpy as np
import pickle


class Attention(nn.Module):
    
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.v = nn.Parameter(torch.rand(hidden_size * 2))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: from decoder, [batch, decoder_hidden_size]
        timestep = encoder_outputs.shape[0]
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)    # [batch, timestep, decoder_hidden_size]
        encoder_outputs = encoder_outputs.transpose(0, 1)    # [batch, timestep, encoder_hidden_size]
        
        # [batch, timestep]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)    # [batch, 1, timestep]
         
    def score(self, hidden, encoder_outputs):
        # hidden: [batch, timestep, decoder_hidden_size]
        # encoder_outputs: [batch, timestep, encoder_hidden_size]
        # energy: [batch, timestep, hidden_size]
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)    # [batch, 2 * hidden_size, timestep]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)    # [batch, 1, 2 * hidden_size]
        energy = torch.bmm(v, energy)    # [batch, 1, timestep]
        return energy.squeeze(1)    # [batch, timestep]
    

class WSeq_attention(nn.Module):

    '''
    Cosine similarity defined in ACL 2017 paper: 
    How to Make Context More Useful?
    An Empirical Study on context-Aware Neural Conversational Models

    mode: sum, concat is very hard to be implemented
    '''

    def __init__(self, mode='sum'):
        super(WSeq_attention, self).__init__()

    def forward(self, query, utterances):
        # query: [batch, hidden], utterances: [seq_len, batch, hidden]
        # cosine similarity
        utterances = utterances.permute(1, 2, 0)    # [batch, hidden, seq_len]
        query = query.reshape(query.shape[0], 1, query.shape[1])    # [batch, 1, hidden]
        p = torch.bmm(query, utterances).squeeze(1)    # [batch, seq_len]
        query_norm = query.squeeze(1).norm(dim=1)    # [batch]
        utterances_norm = utterances.norm(dim=1)    # [batch, seq_len]
        p = p / query_norm.reshape(-1, 1)
        p = p / utterances_norm    # [batch, seq_len]

        # softmax
        sq = torch.ones(p.shape[0], 1)
        if torch.cuda.is_available():
            sq = sq.cuda()
        p = torch.cat([p, sq], 1)    # [batch, seq_len + 1]
        p = F.softmax(p, dim=1)    # [batch, seq_len + 1]

        # mode for getting vector
        utterances = utterances.permute(0, 2, 1)    # [batch, seq_len, hidden]
        vector = torch.cat([utterances, query], 1)   # [batch, seq_len + 1, hidden]
        p = p.unsqueeze(1)    # [batch, 1, seq_len + 1]
        
        # p: [batch, 1, seq_len + 1], vector: [batch, seq_len + 1, hidden]
        vector = torch.bmm(p, vector).squeeze(1)    # [batch, hidden]

        # [batch, hidden]
        return vector
    
class PositionEmbedding(nn.Module):

    '''
    Position embedding for self-attention
    refer: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    d_model: word embedding size or output size of the self-attention blocks
    max_len: the max length of the input squeezec
    '''

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)    # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)    # [1, max_len]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)   # not the parameters of the Module


    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PretrainedEmbedding(nn.Module):

    '''
    Pretrained English BERT contextual word embeddings
    make sure the embedding size is the same as the embed_size setted in the model
    or the error will be thrown.
    '''

    def __init__(self, vocab_size, embed_size, path):
        super(PretrainedEmbedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, embed_size)

        # load pretrained embedding
        with open(path, 'rb') as f:
            emb = pickle.load(f)
        
        self.emb.weight.data.copy_(torch.from_numpy(emb))

    def forward(self, x):
        return self.emb(x)


class NoamOpt:
    '''Optim wrapper that implements rate.'''
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        # Update parameters and rate
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        # Implement `lrate` above
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))



if __name__ == "__main__":
    pass
