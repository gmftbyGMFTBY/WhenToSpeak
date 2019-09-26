#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.9.25


import torch
import torch.nn as nn
import numpy as np
import math
import random
import ipdb
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
            line = [src_w2idx['<sos>']] + [src_w2idx.get(w, src_w2idx['<unk>']) for w in utterance.split()] + [src_w2idx['<eos>']]
            if len(line) > maxlen:
                line = [src_w2idx['<sos>']] + line[-maxlen:]
            turn.append(line)
        src_d.append(turn)

    # tgt
    tgt_d = []
    for example in tgt:
        turn = []
        user, utterance = example[0]
        line = [tgt_w2idx['<sos>']] + [tgt_w2idx.get(w, tgt_w2idx['<unk>']) for w in utterance.split()] + [tgt_w2idx['<eos>']]
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
            line = [vocab['<sos>']] + [vocab.get(w, vocab['<unk>']) for w in utterances.split()] + [vocab['<eos>']]
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


def get_batch_data_cf():
    pass

def get_batch_data_cf_flatten():
    pass


if __name__ == "__main__":
    batch_num = 0
    for sbatch, tbatch, turn_lengths in get_batch_data_flatten('./data/seq2seq/src-train.pkl',
                                                               './data/seq2seq/tgt-train.pkl',
                                                               './processed/seq2seq/iptvocab.pkl',
                                                               './processed/seq2seq/optvocab.pkl', 16, 150):
        print(len(sbatch), tbatch.shape, turn_lengths.shape)
        batch_num += 1
    print(batch_num)
