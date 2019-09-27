#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.9.25

'''
load_data, load_data_flatten can be used by the *_batch_* functions
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
            line = [vocab['<sos>']] + [vocab.get(w, vocab['<unk>']) for w in nltk.word_tokenize(utterances)] + [vocab['<eos>']]
            if len(line) > maxlen:
                line = [vocab['<sos>']] + line[-maxlen:]
            d.append(line)
        return d

    src_dataset = load_(src, src_w2idx)
    tgt_dataset = load_(tgt, tgt_w2idx)

    return src_dataset, tgt_dataset


def get_batch_data_flatten(src, tgt, src_vocab, tgt_vocab, batch_size, maxlen):
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


def get_batch_data(src, tgt, src_vocab, tgt_vocab, batch_size, maxlen):
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
        
        
def load_data_cf(src, tgt, src_vocab, tgt_vocab, maxlen):
    # hierarchical and cf mode
    # return src: (datasize, turns, lengths) & (datasize, turns)[user]
    # return tgt: (datasize, lengths) & (datasize)[user]
    # return label: [datasize,]
    src_w2idx, src_idx2w = load_pickle(src_vocab)
    tgt_w2idx, tgt_idx2w = load_pickle(tgt_vocab)
    user_vocab = ['<0>', '<1>']

    # load data
    src, tgt = load_pickle(src), load_pickle(tgt)
    src_dataset, tgt_dataset, src_user, tgt_user, label = [], [], [], [], []

    # parse
    for sexample, texample in zip(src, tgt):
        # parse src
        turns, srcu = [], []
        for user, utterance in sexample:
            if '<0>' in utterance: user_c = '<0>'
            elif '<1>' in utterance: user_c = '<1>'
            utterance = utterance.replace(user_c, '').strip()
            line = [src_w2idx['<sos>'], src_w2idx[user_c]] + [src_w2idx.get(w, src_w2idx['<unk>']) for w in nltk.word_tokenize(utterance)] + [src_w2idx['<eos>']]
            if len(line) > maxlen:
                line = [src_w2idx['<sos>'], line[1]] + line[-maxlen:]
            turns.append(line)
            srcu.append(user_vocab.index(user))

        # parse tgt
        tgtu, utterance = texample[0]
        if '<0>' in utterance: user_c = '<0>'
        elif '<1>' in utterance: user_c = '<1>'
        utterance = utterance.replace(user_c, '').strip()
        tgtu = user_vocab.index(tgtu)
        line = [tgt_w2idx['<sos>'], tgt_w2idx[user_c]] + [tgt_w2idx.get(w, tgt_w2idx['<unk>']) for w in nltk.word_tokenize(utterance)] + [tgt_w2idx['<eos>']]
        if len(line) > maxlen:
            line = [tgt_w2idx['<sos>'], line[1]] + line[-maxlen:]

        label.append(1)
        src_dataset.append(turns)
        tgt_dataset.append(line)
        src_user.append(srcu)
        tgt_user.append(tgtu)
        
        if tgtu != srcu[-1]:
            # two parts
            # 1. model is srcu, label 0, utterance add silence
            label.append(0)
            src_dataset.append(turns)
            tgt_dataset.append([tgt_w2idx['<sos>'], tgt_w2idx['<silence>'], tgt_w2idx['<eos>']])
            src_user.append(srcu)
            tgt_user.append(srcu[-1])
            # 2. model is tgtu, label 1
    return src_dataset, src_user, tgt_dataset, tgt_user, label


def get_batch_data_cf(src, tgt, src_vocab, tgt_vocab, batch_size, maxlen):
    '''get batch data of [cf & hierarchical]
    return data:
    - sbatch: [turn, batch, length]
    - tbatch: [batch, length]
    - turn_lengths: [batch]
    - de: [batch]
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



if __name__ == "__main__":
    # batch_num = 0
    # for sbatch, tbatch, turn_lengths in get_batch_data_flatten('./data/seq2seq/src-train.pkl',
    #                                                            './data/seq2seq/tgt-train.pkl',
    #                                                            './processed/seq2seq/iptvocab.pkl',
    #                                                            './processed/seq2seq/optvocab.pkl', 16, 150):
    #     print(len(sbatch), tbatch.shape, turn_lengths.shape)
    #     batch_num += 1
    # print(batch_num)

    batch_num = 0
    for sbatch, tbatch, subatch, tubatch, lbatch, turn_lengths in get_batch_data_cf('./data/seq2seq/cf/src-train.pkl', './data/seq2seq/cf/tgt-train.pkl', './processed/seq2seq/iptvocab.pkl', './processed/seq2seq/optvocab.pkl', 32, 50):
        ipdb.set_trace()
        batch_num += 1
    print(batch_num)
