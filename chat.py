#!/usr/bin/python
# Author: GMFTBY
# Time: 2019.11.8


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
from model.when2talk import When2Talk
from model.GCNRNN import GCNRNN
from model.GatedGCN import GatedGCN
from model.W2T_RNN_First import W2T_RNN_First
from model.W2T_GCNRNN import W2T_GCNRNN
from model.GatedGCN_nobi import GatedGCN_nobi
from model.GATRNN import GATRNN

class Bot:
    
    def __init__(self, args):
        # load vocab
        tgt_vocab = load_pickle(kwargs['tgt_vocab'])
        src_vocab = load_pickle(kwargs['src_vocab'])
        self.src_w2idx, self.src_idx2w = src_vocab
        self.tgt_w2idx, self.tgt_idx2w = tgt_vocab
        
        # load model
        # load net
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
        elif kwargs['model'] == 'when2talk':
            net = When2Talk(len(src_w2idx), len(tgt_w2idx), kwargs['embed_size'],
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
            
        self.net = net
        if torch.cuda.is_available():
            self.net.cuda()
        self.net.eval()
        
        print('Net:')
        print(net)
        
        # load checkpoint
        # load best model
        load_best_model(kwargs["dataset"], kwargs['model'], self.net, 
                        kwargs['min_threshold'], kwargs['max_threshold'])
        
    def process_input():
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate script')
    parser.add_argument('--src_test', type=str, default=None, help='src test file')
    parser.add_argument('--tgt_test', type=str, default=None, help='tgt test file')
    parser.add_argument('--min_threshold', type=int, default=20, 
                        help='epoch threshold for loading best model')
    parser.add_argument('--max_threshold', type=int, default=20, 
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
    
    chatbot = Bot(args)