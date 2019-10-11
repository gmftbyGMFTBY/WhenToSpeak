#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.9.25

'''
data loader of [flatten, hierarchical], [cf, ncf], [GCN, unGCN] mode
'''

import torch
import torch.nn as nn
import numpy as np
import math
import random
import ipdb
import nltk
from utils import * 


def load_data(src, tgt, src_vocab, tgt_vocab, maxlen):
    # sort by the lengths
    # convert src data from [datasize, (user, utterance)] into [datasize, turns, lengths]
    # convert tgt data from [datasize, (user, utterance)] into [datasize, length]
    src_w2idx, src_idx2w = load_pickle(src_vocab)
    tgt_w2idx, tgt_idx2w = load_pickle(tgt_vocab)

    src, tgt = load_pickle(src), load_pickle(tgt)
    # src
    src_d = []
    for example in src:
        turn = []
        for user, utterance in example:
            if '<0>' in utterance: user_c = '<0>'
            elif '<1>' in utterance: user_c = '<1>'
            utterance = utterance.replace(user_c, '').strip()
            line = [src_w2idx['<sos>'], src_w2idx[user_c]] + [src_w2idx.get(w, src_w2idx['<unk>']) for w in nltk.word_tokenize(utterance)] + [src_w2idx['<eos>']]
            if len(line) > maxlen:
                line = [src_w2idx['<sos>']] + line[-maxlen:]
            turn.append(line)
        src_d.append(turn)

    # tgt
    tgt_d = []
    for example in tgt:
        turn = []
        user, utterance = example[0]
        if '<0>' in utterance: user_c = '<0>'
        elif '<1>' in utterance: user_c = '<1>'
        utterance = utterance.replace(user_c, '').strip()
        line = [tgt_w2idx['<sos>'], tgt_w2idx[user_c]] + [tgt_w2idx.get(w, tgt_w2idx['<unk>']) for w in nltk.word_tokenize(utterance)] + [tgt_w2idx['<eos>']]
        if len(line) > maxlen:
            line = [tgt_w2idx['<sos>']] + line[-maxlen:]
        tgt_d.append(line)

    return src_d, tgt_d


def load_data_flatten(src, tgt, src_vocab, tgt_vocab, maxlen):
    # return [datasize, length]
    src_w2idx, src_idx2w = load_pickle(src_vocab)
    tgt_w2idx, tgt_idx2w = load_pickle(tgt_vocab)
    src, tgt = load_pickle(src), load_pickle(tgt)

    def load_(data, vocab):
        d = []
        for example in data:
            utterances = ' <eou> '.join([i[1] for i in example])
            utterances = utterances.replace('<0>', '').strip()
            utterances = utterances.replace('<1>', '').strip()
            line = [vocab['<sos>']] + [vocab.get(w, vocab['<unk>']) for w in nltk.word_tokenize(utterances)] + [vocab['<eos>']]
            if len(line) > maxlen:
                line = [vocab['<sos>']] + line[-maxlen:]
            d.append(line)
        return d

    src_dataset = load_(src, src_w2idx)
    tgt_dataset = load_(tgt, tgt_w2idx)

    return src_dataset, tgt_dataset


def get_batch_data_flatten(src, tgt, src_vocab, tgt_vocab, batch_size, maxlen, plus=0):
    # flatten mode donot need plus
    src_w2idx, src_idx2w = load_pickle(src_vocab)
    tgt_w2idx, tgt_idx2w = load_pickle(tgt_vocab)

    src_dataset, tgt_dataset = load_data_flatten(src, tgt, src_vocab, tgt_vocab, maxlen)
    turns = [len(dialog) for dialog in src_dataset]
    turnidx = np.argsort(turns)
    # sort by the length of the turns
    src_dataset = [src_dataset[idx] for idx in turnidx]
    tgt_dataset = [tgt_dataset[idx] for idx in turnidx]

    # batch convert to tensor
    fidx, bidx = 0, 0
    while fidx < len(src_dataset):
        bidx = fidx + batch_size
        sbatch, tbatch = src_dataset[fidx:bidx], tgt_dataset[fidx:bidx]
        # shuffle
        shuffleidx = np.arange(0, len(sbatch))
        np.random.shuffle(shuffleidx)
        sbatch = [sbatch[idx] for idx in shuffleidx]
        tbatch = [tbatch[idx] for idx in shuffleidx]

        bs = len(sbatch)

        # pad sbatch and tbatch
        turn_lengths = [len(sbatch[i]) for i in range(bs)]
        pad_sequence(src_w2idx['<pad>'], sbatch, bs)
        pad_sequence(tgt_w2idx['<pad>'], tbatch, bs)

        # [seq_len, batch]
        sbatch = torch.tensor(sbatch, dtype=torch.long).transpose(0, 1)
        tbatch = torch.tensor(tbatch, dtype=torch.long).transpose(0, 1)
        turn_lengths = torch.tensor(turn_lengths, dtype=torch.long)
        if torch.cuda.is_available():
            tbatch = tbatch.cuda()
            sbatch = sbatch.cuda()
            turn_lengths = turn_lengths.cuda()

        fidx = bidx

        yield sbatch, tbatch, turn_lengths


def get_batch_data(src, tgt, src_vocab, tgt_vocab, batch_size, maxlen, plus=0):
    src_w2idx, src_idx2w = load_pickle(src_vocab)
    tgt_w2idx, tgt_idx2w = load_pickle(tgt_vocab)

    src_dataset, tgt_dataset = load_data(src, tgt, src_vocab, tgt_vocab, maxlen)
    turns = [len(dialog) for dialog in src_dataset]
    turnidx = np.argsort(turns)

    src_dataset = [src_dataset[idx] for idx in turnidx]
    tgt_dataset = [tgt_dataset[idx] for idx in turnidx]

    turns = [len(dialog) for dialog in src_dataset]
    fidx, bidx = 0, 0
    while fidx < len(src_dataset):
        bidx = fidx + batch_size
        head = turns[fidx]
        cidx = 10000
        for p, i in enumerate(turns[fidx:bidx]):
            if i != head:
                cidx = p
                break
        cidx = fidx + cidx
        bidx = min(bidx, cidx)

        sbatch, tbatch = src_dataset[fidx:bidx], tgt_dataset[fidx:bidx]

        if len(sbatch[0]) < plus:
            fidx = bidx
            continue
            
            
        # delete 3/4 low-turns dataset for training
        # if plus < 0 and len(sbatch[0]) < abs(plus):
        #     if random.random() > 0.25:
        #         fidx = bidx
        #         continue
                

        shuffleidx = np.arange(0, len(sbatch))
        np.random.shuffle(shuffleidx)
        sbatch = [sbatch[idx] for idx in shuffleidx]
        tbatch = [tbatch[idx] for idx in shuffleidx]

        sbatch = transformer_list(sbatch)    # [turns, batch, lengths]
        bs, ts = len(sbatch[0]), len(sbatch)

        turn_lengths = []
        for i in range(ts):
            lengths = []
            for item in sbatch[i]:
                lengths.append(len(item))
            turn_lengths.append(lengths)
            pad_sequence(src_w2idx['<pad>'], sbatch[i], bs)

        pad_sequence(tgt_w2idx['<pad>'], tbatch, bs)

        srcbatch = []
        for i in range(ts):
            pause = torch.tensor(sbatch[i], dtype=torch.long).transpose(0, 1)
            if torch.cuda.is_available():
                pause = pause.cuda()
            srcbatch.append(pause)
        sbatch = srcbatch
        tbatch = torch.tensor(tbatch, dtype=torch.long).transpose(0, 1)
        if torch.cuda.is_available():
            tbatch = tbatch.cuda()

        turn_lengths = torch.tensor(turn_lengths, dtype=torch.long)
        if torch.cuda.is_available():
            turn_lengths = turn_lengths.cuda()

        fidx = bidx

        yield sbatch, tbatch, turn_lengths


def get_batch_data_cf(src, tgt, src_vocab, tgt_vocab, batch_size, maxlen, plus=0):
    '''get batch data of [cf & hierarchical]
    return data:
    - sbatch: [turn, batch, length]
    - tbatch: [batch, length]
    - subatch: [batch], src user
    - tubatch: [batch], tgt user
    - label: [batch], speaking timing
    - turn_lengths: [batch]
    '''
    src_w2idx, src_idx2w = load_pickle(src_vocab)
    tgt_w2idx, tgt_idx2w = load_pickle(tgt_vocab)

    src_dataset, src_user, tgt_dataset, tgt_user, label = load_data_cf(src, tgt, src_vocab, tgt_vocab, maxlen)
    turns = [len(dialog) for dialog in src_dataset]
    turnidx = np.argsort(turns)

    # [datasize, turn, lengths]
    src_dataset = [src_dataset[idx] for idx in turnidx]
    tgt_dataset = [tgt_dataset[idx] for idx in turnidx]
    src_user = [src_user[idx] for idx in turnidx]
    tgt_user = [tgt_user[idx] for idx in turnidx]
    label = [label[idx] for idx in turnidx]
    
    turns = [len(dialog) for dialog in src_dataset]
    fidx, bidx = 0, 0
    while fidx < len(src_dataset):
        bidx = fidx + batch_size
        head = turns[fidx]
        cidx = 10000
        for p, i in enumerate(turns[fidx:bidx]):
            if i != head:
                cidx = p
                break
        cidx = fidx + cidx
        bidx = min(bidx, cidx)

        sbatch, tbatch, subatch, tubatch, lbatch = src_dataset[fidx:bidx], tgt_dataset[fidx:bidx], src_user[fidx:bidx], tgt_user[fidx:bidx], label[fidx:bidx]

        if len(sbatch[0]) < plus:
            fidx = bidx
            continue
            
            
        # delete 3/4 low-turns dataset for training
        # if plus < 0 and len(sbatch[0]) < abs(plus):
        #     if random.random() > 0.25:
        #         fidx = bidx
        #         continue
                
        
        shuffleidx = np.arange(0, len(sbatch))
        np.random.shuffle(shuffleidx)
        sbatch = [sbatch[idx] for idx in shuffleidx]   # [batch, turns, lengths]
        tbatch = [tbatch[idx] for idx in shuffleidx]   # [batch, lengths]
        subatch = [subatch[idx] for idx in shuffleidx]    # [batch, turns]
        tubatch = [tubatch[idx] for idx in shuffleidx]    # [batch,]
        lbatch = [lbatch[idx] for idx in shuffleidx]      # [batch,]

        sbatch = transformer_list(sbatch)    # [turns, batch, lengths]
        bs, ts = len(sbatch[0]), len(sbatch)

        turn_lengths = []
        for i in range(ts):
            lengths = []
            for item in sbatch[i]:
                lengths.append(len(item))
            turn_lengths.append(lengths)
            pad_sequence(src_w2idx['<pad>'], sbatch[i], bs)

        pad_sequence(tgt_w2idx['<pad>'], tbatch, bs)

        # convert to tensor
        srcbatch = []
        for i in range(ts):
            pause = torch.tensor(sbatch[i], dtype=torch.long).transpose(0, 1)
            if torch.cuda.is_available():
                pause = pause.cuda()
            srcbatch.append(pause)    # [turns, seq_len, batch]
        sbatch = srcbatch
        tbatch = torch.tensor(tbatch, dtype=torch.long).transpose(0, 1)    # [seq_len, batch]
        subatch = torch.tensor(subatch, dtype=torch.long).transpose(0, 1)   # [turns, batch]
        tubatch = torch.tensor(tubatch, dtype=torch.long)    # [batch]
        lbatch = torch.tensor(lbatch, dtype=torch.float)      # [batch]
        turn_lengths = torch.tensor(turn_lengths, dtype=torch.long)     # [batch]
        if torch.cuda.is_available():
            tbatch = tbatch.cuda()
            subatch = subatch.cuda()
            tubatch = tubatch.cuda()
            lbatch = lbatch.cuda()
            turn_lengths = turn_lengths.cuda()
        
        fidx = bidx
        yield sbatch, tbatch, subatch, tubatch, lbatch, turn_lengths
        
        
def get_batch_data_cf_graph(src, tgt, graph, src_vocab, tgt_vocab, batch_size, maxlen, plus=0):
    '''get batch data of [cf & hierarchical & graph]
    return data:
    - sbatch: [turn, batch, length]
    - tbatch: [batch, length]
    - gbatch: [batch, ([2, num_edge], [num_edge])]
    - subatch: [batch], src user
    - tubatch: [batch], tgt user
    - label: [batch], speaking timing
    - turn_lengths: [batch]
    '''
    src_w2idx, src_idx2w = load_pickle(src_vocab)
    tgt_w2idx, tgt_idx2w = load_pickle(tgt_vocab)

    src_dataset, src_user, tgt_dataset, tgt_user, label = load_data_cf(src, tgt, src_vocab, tgt_vocab, maxlen)
    graph = load_pickle(graph)    # [datasize, (edges, weight)]
    
    turns = [len(dialog) for dialog in src_dataset]
    
    # prune the dataset before the shuffle processing
    
    turnidx = np.argsort(turns)

    # [datasize, turn, lengths]
    src_dataset = [src_dataset[idx] for idx in turnidx]
    tgt_dataset = [tgt_dataset[idx] for idx in turnidx]
    graph = [graph[idx] for idx in turnidx]
    src_user = [src_user[idx] for idx in turnidx]
    tgt_user = [tgt_user[idx] for idx in turnidx]
    label = [label[idx] for idx in turnidx]
    
    turns = [len(dialog) for dialog in src_dataset]
    fidx, bidx = 0, 0
    while fidx < len(src_dataset):
        bidx = fidx + batch_size
        head = turns[fidx]
        cidx = 10000
        for p, i in enumerate(turns[fidx:bidx]):
            if i != head:
                cidx = p
                break
        cidx = fidx + cidx
        bidx = min(bidx, cidx)

        sbatch, tbatch, subatch, tubatch, lbatch = src_dataset[fidx:bidx], tgt_dataset[fidx:bidx], src_user[fidx:bidx], tgt_user[fidx:bidx], label[fidx:bidx]
        gbatch = graph[fidx:bidx]

        # check the turn size for the plus experiment mode
        if len(sbatch[0]) < plus:
            fidx = bidx
            continue
            
            
        # delete 3/4 low-turns dataset for training
        # if plus < 0 and len(sbatch[0]) < abs(plus):
        #     if random.random() > 0.25:
        #         fidx = bidx
        #         continue
            
        
        shuffleidx = np.arange(0, len(sbatch))
        np.random.shuffle(shuffleidx)
        sbatch = [sbatch[idx] for idx in shuffleidx]   # [batch, turns, lengths]
        tbatch = [tbatch[idx] for idx in shuffleidx]   # [batch, lengths]
        subatch = [subatch[idx] for idx in shuffleidx]    # [batch, turns]
        tubatch = [tubatch[idx] for idx in shuffleidx]    # [batch,]
        lbatch = [lbatch[idx] for idx in shuffleidx]      # [batch,]
        gbatch = [gbatch[idx] for idx in shuffleidx]      # [batch, ([2, edges_num], [edges_num]),]
        
        sbatch = transformer_list(sbatch)    # [turns, batch, lengths]
        bs, ts = len(sbatch[0]), len(sbatch)
        
        # gbatch can be converted to a DataLoader which only hold one batch
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html, FAQ(2)
        # the dataloader will be created in the model/when2talk.py 
        # because of it needs the hidden state of the utterance encode as the node features, here we only need to return the gbatch

        turn_lengths = []
        for i in range(ts):
            lengths = []
            for item in sbatch[i]:
                lengths.append(len(item))
            turn_lengths.append(lengths)
            pad_sequence(src_w2idx['<pad>'], sbatch[i], bs)

        pad_sequence(tgt_w2idx['<pad>'], tbatch, bs)

        # convert to tensor
        srcbatch = []
        for i in range(ts):
            pause = torch.tensor(sbatch[i], dtype=torch.long).transpose(0, 1)
            if torch.cuda.is_available():
                pause = pause.cuda()
            srcbatch.append(pause)    # [turns, seq_len, batch]
        sbatch = srcbatch
        tbatch = torch.tensor(tbatch, dtype=torch.long).transpose(0, 1)    # [seq_len, batch]
        subatch = torch.tensor(subatch, dtype=torch.long).transpose(0, 1)   # [turns, batch]
        tubatch = torch.tensor(tubatch, dtype=torch.long)    # [batch]
        lbatch = torch.tensor(lbatch, dtype=torch.float)      # [batch]
        turn_lengths = torch.tensor(turn_lengths, dtype=torch.long)     # [batch]
        if torch.cuda.is_available():
            tbatch = tbatch.cuda()
            subatch = subatch.cuda()
            tubatch = tubatch.cuda()
            lbatch = lbatch.cuda()
            turn_lengths = turn_lengths.cuda()
        
        fidx = bidx
        yield sbatch, tbatch, gbatch, subatch, tubatch, lbatch, turn_lengths
        

        
if __name__ == "__main__":
    # batch_num = 0
    # for sbatch, tbatch, turn_lengths in get_batch_data_flatten('./data/cornell-corpus/ncf/src-train.pkl',
    #                                                            './data/cornell-corpus/ncf/tgt-train.pkl',
    #                                                            './processed/cornell/iptvocab.pkl',
    #                                                            './processed/cornell/optvocab.pkl', 32, 150):
    #     print(len(sbatch), tbatch.shape, turn_lengths.shape)
    #     batch_num += 1
    # print(batch_num)

    batch_num, zero, one = 0, 0, 0
    # stat the ratio of the 0 label in the cf mode dataset
    for sbatch, tbatch, gbatch, subatch, tubatch, lbatch, turn_lengths in get_batch_data_cf_graph('./data/dailydialog-corpus/cf/src-train.pkl', 
                                                                                    './data/dailydialog-corpus/cf/tgt-train.pkl',
                                                                                    './processed/dailydialog/train-graph.pkl',
                                                                                    './processed/dailydialog/iptvocab.pkl',
                                                                                    './processed/dailydialog/optvocab.pkl', 32, 50):
        if batch_num == 6719:
            ipdb.set_trace()
        batch_num += 1
        o = torch.sum(lbatch).item()
        one += o
        zero += len(lbatch) - o
    print(batch_num, zero, one)
