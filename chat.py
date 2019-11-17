#!/usr/bin/python
# Author: GMFTBY
# Time: 2019.11.8


'''
Chat script, show the demo
'''


import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import math
import argparse

from utils import *
from data_loader import *
from model.seq2seq_attention import Seq2Seq
from model.HRED import HRED
from model.HRED_cf import HRED_cf
from model.when2talk_GCN import When2Talk_GCN
from model.when2talk_GAT import When2Talk_GAT
from model.GCNRNN import GCNRNN
from model.GatedGCN import GatedGCN
from model.W2T_RNN_First import W2T_RNN_First
from model.W2T_GCNRNN import W2T_GCNRNN
from model.GatedGCN_nobi import GatedGCN_nobi
from model.GATRNN import GATRNN


def create_model(kwargs, src_w2idx, tgt_w2idx):
    # load model
    # load net
    kwargs = vars(kwargs)
    if kwargs['model'] == 'seq2seq':
        net = Seq2Seq(len(src_w2idx), kwargs['embed_size'], 
                      len(tgt_w2idx), kwargs['utter_hidden'], 
                      kwargs['decoder_hidden'], pad=tgt_w2idx['<pad>'], 
                      sos=tgt_w2idx['<sos>'], utter_n_layer=kwargs['utter_n_layer'])
    elif kwargs['model'] == 'hred':
        net = HRED(kwargs['embed_size'], len(src_w2idx), len(tgt_w2idx),
                   kwargs['utter_hidden'], kwargs['context_hidden'],
                   kwargs['decoder_hidden'], pad=tgt_w2idx['<pad>'],
                   sos=tgt_w2idx['<sos>'], utter_n_layer=kwargs['utter_n_layer'])
    elif kwargs['model'] == 'hred-cf':
        net = HRED_cf(kwargs['embed_size'], len(src_w2idx), len(tgt_w2idx),
                      kwargs['utter_hidden'], kwargs['context_hidden'],
                      kwargs['decoder_hidden'], pad=tgt_w2idx['<pad>'],
                      sos=tgt_w2idx['<sos>'], utter_n_layer=kwargs['utter_n_layer'],
                      user_embed_size=kwargs['user_embed_size'])
    elif kwargs['model'] == 'when2talk_GCN':
        net = When2Talk_GCN(len(src_w2idx), len(tgt_w2idx), kwargs['embed_size'],
                            kwargs['utter_hidden'], kwargs['context_hidden'],
                            kwargs['decoder_hidden'], kwargs['position_embed_size'], 
                            user_embed_size=kwargs['user_embed_size'],
                            sos=tgt_w2idx["<sos>"], pad=tgt_w2idx['<pad>'], 
                            utter_n_layer=kwargs['utter_n_layer'],
                            contextrnn=kwargs['contextrnn'])
    elif kwargs['model'] == 'when2talk_GAT':
        net = When2Talk_GAT(len(src_w2idx), len(tgt_w2idx), kwargs['embed_size'],
                            kwargs['utter_hidden'], kwargs['context_hidden'],
                            kwargs['decoder_hidden'], kwargs['position_embed_size'], 
                            user_embed_size=kwargs['user_embed_size'],
                            sos=tgt_w2idx["<sos>"], pad=tgt_w2idx['<pad>'], 
                            utter_n_layer=kwargs['utter_n_layer'],
                            contextrnn=kwargs['contextrnn'])
    elif kwargs['model'] == 'GATRNN':
        net = GATRNN(len(src_w2idx), len(tgt_w2idx), kwargs['embed_size'],
                     kwargs['utter_hidden'], kwargs['context_hidden'],
                     kwargs['decoder_hidden'], kwargs['position_embed_size'], 
                     user_embed_size=kwargs['user_embed_size'],
                     sos=tgt_w2idx["<sos>"], pad=tgt_w2idx['<pad>'], 
                     utter_n_layer=kwargs['utter_n_layer'],
                     context_threshold=kwargs['context_threshold'])
    elif kwargs['model'] == 'GCNRNN':
        net = GCNRNN(len(src_w2idx), len(tgt_w2idx), kwargs['embed_size'],
                     kwargs['utter_hidden'], kwargs['context_hidden'],
                     kwargs['decoder_hidden'], kwargs['position_embed_size'], 
                     user_embed_size=kwargs['user_embed_size'],
                     sos=tgt_w2idx["<sos>"], pad=tgt_w2idx['<pad>'], 
                     utter_n_layer=kwargs['utter_n_layer'],
                     context_threshold=kwargs['context_threshold'])
    elif kwargs['model'] == 'W2T_GCNRNN':
        net = W2T_GCNRNN(len(src_w2idx), len(tgt_w2idx),
                         kwargs['embed_size'],
                         kwargs['utter_hidden'],
                         kwargs['context_hidden'],
                         kwargs['decoder_hidden'], 
                         kwargs['position_embed_size'],
                         user_embed_size=kwargs['user_embed_size'],
                         sos=tgt_w2idx["<sos>"],
                         pad=tgt_w2idx['<pad>'],
                         utter_n_layer=kwargs['utter_n_layer'])
    elif kwargs['model'] == 'GatedGCN':
        net = GatedGCN(len(src_w2idx), len(tgt_w2idx), kwargs['embed_size'],
                       kwargs['utter_hidden'], kwargs['context_hidden'],
                       kwargs['decoder_hidden'], kwargs['position_embed_size'], 
                       user_embed_size=kwargs['user_embed_size'],
                       sos=tgt_w2idx["<sos>"], pad=tgt_w2idx['<pad>'], 
                       utter_n_layer=kwargs['utter_n_layer'],
                       context_threshold=kwargs['context_threshold'])
    elif kwargs['model'] == 'GatedGCN_nobi':
        net = GatedGCN_nobi(len(src_w2idx), len(tgt_w2idx), kwargs['embed_size'],
                            kwargs['utter_hidden'], kwargs['context_hidden'],
                            kwargs['decoder_hidden'], kwargs['position_embed_size'], 
                            user_embed_size=kwargs['user_embed_size'],
                            sos=tgt_w2idx["<sos>"], pad=tgt_w2idx['<pad>'], 
                            utter_n_layer=kwargs['utter_n_layer'],
                            context_threshold=kwargs['context_threshold'])
    elif kwargs['model'] == 'W2T_RNN_First':
        net = W2T_RNN_First(len(src_w2idx), len(tgt_w2idx), kwargs['embed_size'],
                            kwargs['utter_hidden'], kwargs['context_hidden'],
                            kwargs['decoder_hidden'], kwargs['position_embed_size'], 
                            user_embed_size=kwargs['user_embed_size'],
                            sos=tgt_w2idx["<sos>"], pad=tgt_w2idx['<pad>'], 
                            utter_n_layer=kwargs['utter_n_layer'])
    else:
        raise Exception('[!] wrong model (seq2seq, hred, hred-cf)')
        
    return net


class Bot:
    
    def __init__(self, kwargs, maxlen=50, role='<1>'):
        # load vocab
        tgt_vocab = load_pickle(kwargs.tgt_vocab)
        src_vocab = load_pickle(kwargs.src_vocab)
        self.src_w2idx, self.src_idx2w = src_vocab
        self.tgt_w2idx, self.tgt_idx2w = tgt_vocab
        
        # whether have the ability to decide the talk timing
        if args.model in ['hred', 'seq2seq']:
            self.decision = False
        else:
            self.decision = True
            
        # load the model
        self.net = create_model(args, self.src_w2idx, self.tgt_w2idx)
        if torch.cuda.is_available():
            self.net.cuda()
        self.net.eval()
        print('Net:')
        print(self.net)
        
        # load checkpoint
        load_best_model(args.dataset, args.model, self.net, 
                        args.min_threshold, args.max_threshold)
        
        # reset flag
        self.reset = True
        self.container = []
        self.history = []
        self.roles = ['<0>', '<1>']    # <0>: human, <1>: chatbot
        self.role = role
        
        # print configure
        self.maxlen = max(50, maxlen)
        self.src_maxlen = 100
        
        print('[!] Init the model over')
        
    def str2tensor(self, utterance, role):
        line = [self.src_w2idx['<sos>'], self.src_w2idx[role]] + [self.src_w2idx.get(w, self.src_w2idx['<unk>']) for w in nltk.word_tokenize(utterance)] + [self.src_w2idx['<eos>']]
        if len(line) > self.src_maxlen:
            line = [self.src_w2idx['<sos>'], line[1]] + line[-maxlen:]
        return line
    
    def get_role(self, role):
        try:
            role = self.roles.index(role)
        except:
            raise Exception(f'[!] Unknown role {role}')
        return role
    
    def add_sentence(self, utterance, role):
        self.history.append((role, utterance))
        nrole = self.get_role(role)
        self.container.append((nrole, self.str2tensor(utterance, role)))
        
    def create_graph(self):
        # create the graph by using self.container
        # role information and temporal information
        edges = {}
        turn_len = len(self.container)
        # temporal information
        for i in range(turn_len - 1):
            edges[(i, i + 1)] = [1]
            
        # role information
        for i in range(turn_len):
            for j in range(turn_len):
                if j > i:
                    useri, _ = self.container[i]
                    userj, _ = self.container[j]
                    if useri == userj:
                        if edges.get((i, j), None):
                            edges[(i, j)].append(1)
                        else:
                            edges[(i, j)] = [1]
                            
        # clear
        e, w = [[], []], []
        for src, tgt in edges.keys():
            e[0].append(src)
            e[1].append(tgt)
            w.append(max(edges[(src, tgt)]))
        return (e, w)
        
    def process_input(self):
        '''role: chatbot / human'''
        # add to the container
        # self.add_sentence(utterance, role)
        
        # generate the graph
        gbatch = [self.create_graph()]
        
        # src_utterance, src_role
        sbatch, subatch = [], []
        for i in self.container:
            sbatch.append(self.load2GPU(torch.tensor(i[1], dtype=torch.long).unsqueeze(1)))
            subatch.append(i[0])
        subatch = self.load2GPU(torch.tensor(subatch, dtype=torch.long).unsqueeze(1))
        
        # tubatch
        tubatch = self.load2GPU(torch.tensor([self.get_role(self.role)], dtype=torch.long))
        
        # turn_lengths
        turn_lengths = [[len(i[1])] for i in self.container]
        turn_lengths = self.load2GPU(torch.tensor(turn_lengths, dtype=torch.long))
        
        return sbatch, gbatch, subatch, tubatch, self.maxlen, turn_lengths
    
    def load2GPU(self, t):
        if torch.cuda.is_available():
            t = t.cuda()
        return t
    
    def tensor2str(self, t):
        rest = []
        for i in t[1:]:
            w = self.tgt_idx2w[i]
            if w in ['<pad>', '<eos>']:
                break
            rest.append(w)
        return ' '.join(rest)
    
    def generate(self):
        sbatch, gbatch, subatch, tubatch, maxlen, turn_lengths = self.process_input()
        # de: [1], outputs: [maxlen, 1]
        de, output = self.net.predict(sbatch, gbatch, subatch, tubatch, maxlen, turn_lengths)
        output = list(map(int, output.squeeze(1).cpu().tolist()))     # [maxlen]
        de = de.cpu().item() > 0.5
        if de:
            # Talk
            return self.tensor2str(output)
        else:
            return '<silence>'
        
    def show_history(self):
        for i in self.history:
            print(f'{i[0]}: {i[1]}')
    
    def set_reset(self):
        self.container = []
        self.history = []
        self.reset = True
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate script')
    parser.add_argument('--src_test', type=str, default=None, help='src test file')
    parser.add_argument('--tgt_test', type=str, default=None, help='tgt test file')
    parser.add_argument('--min_threshold', type=int, default=0, 
                        help='epoch threshold for loading best model')
    parser.add_argument('--max_threshold', type=int, default=30, 
                        help='epoch threshold for loading best model')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--model', type=str, default='HRED', help='model to be trained')
    parser.add_argument('--utter_n_layer', type=int, default=1, help='layer of encoder')
    parser.add_argument('--utter_hidden', type=int, default=150, 
                        help='utterance encoder hidden size')
    parser.add_argument('--context_hidden', type=int, default=150, 
                        help='context encoder hidden size')
    parser.add_argument('--decoder_hidden', type=int, default=150, 
                        help='decoder hidden size')
    parser.add_argument('--seed', type=int, default=30,
                        help='random seed')
    parser.add_argument('--embed_size', type=int, default=200, 
                        help='embedding layer size')
    parser.add_argument('--src_vocab', type=str, default=None, help='src vocabulary')
    parser.add_argument('--tgt_vocab', type=str, default=None, help='tgt vocabulary')
    parser.add_argument('--maxlen', type=int, default=50, help='the maxlen of the utterance')
    parser.add_argument('--pred', type=str, default=None, 
                        help='the csv file save the output')
    parser.add_argument('--hierarchical', type=int, default=1, help='whether hierarchical architecture')
    parser.add_argument('--tgt_maxlen', type=int, default=50, help='target sequence maxlen')
    parser.add_argument('--user_embed_size', type=int, default=10, help='user embed size')
    parser.add_argument('--cf', type=int, default=0, help='whether have the classification')
    parser.add_argument('--dataset', type=str, default='ubuntu')
    parser.add_argument('--position_embed_size', type=int, default=30)
    parser.add_argument('--graph', type=int, default=0)
    parser.add_argument('--test_graph', type=str, default=None)
    parser.add_argument('--plus', type=int, default=0, help='the same as the one in train.py')
    parser.add_argument('--contextrnn', dest='contextrnn', action='store_true')
    parser.add_argument('--no-contextrnn', dest='contextrnn', action='store_false')
    parser.add_argument('--context_threshold', type=int, default=2)

    args = parser.parse_args()
    
    # show the parameters
    print('Parameters:')
    print(args)
    
    chatbot = Bot(args, maxlen=args.maxlen, role='<1>')
    
    # begin to chat with human
    for i in range(100):
        print(f'===== Dialogue {i} begin =====')
        while True:
            utterance = input(f'<0>: ')
            utterance = utterance.strip()
            if 'exit' in utterance:
                break
            chatbot.add_sentence(utterance, '<0>')
            
            while True:
                response = chatbot.generate()
                if 'silence' in response:
                    break
                else:
                    response = response.replace('<1>', '').replace('<0>', '').strip()
                    chatbot.add_sentence(response, '<1>')
                    print(f'<1> {response}')
        
        print(f'===== Dialogue {i} finish =====')
        chatbot.show_history()
        chatbot.set_reset()