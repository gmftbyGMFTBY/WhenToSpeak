#!/usr/bin/python
# Author: GMFTBY
# Time: 2019.9.25

'''
utils functions for training the model
'''


import pickle
import argparse
from collections import Counter
import os
import numpy as np
import torch
import nltk


def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def num2seq(src, idx2w):
    return [idx2w[int(i)] for i in src]


def transformer_list(obj):
    # transform [batch, turn, lengths] to [turns, batch, lengths]
    turns = []
    batch_size, turn_size = len(obj), len(obj[0])
    for i in range(turn_size):
        turns.append([obj[j][i] for j in range(batch_size)])
    return turns


def pad_sequence(pad, batch, bs):
    maxlen = max([len(batch[i]) for i in range(bs)])
    for i in range(bs):
        batch[i].extend([pad] * (maxlen - len(batch[i])))


def load_best_model(model, net, threshold):
    path = f'./ckpt/{model}/'
    best_loss, best_file, best_epoch = np.inf, None, -1

    for file in os.listdir(path):
        _, val_loss, _, epoch = file.split('_')
        epoch = epoch.split('.')[0]
        val_loss, epoch = float(val_loss), int(epoch)

        if epoch >= threshold and val_loss < best_loss:
            best_file = file
            best_loss = val_loss

    if best_file:
        file_path = path + best_file
        print(f'[!] Load the model from {file_path}, threshold {threshold}')
        net.load_state_dict(torch.load(file_path)['net'])
    else:
        raise Exception('[!] No saved model')
        
        
def generate_vocab(files, vocab, cutoff=30000):
    # training and validation files
    words = []
    for file in files:
        obj = load_pickle(file)
        for example in obj:
            for turn in example:
                user, utterance = turn
                utterance = utterance.replace('<0>', '')
                utterance = utterance.replace('<1>', '')
                words.extend(nltk.word_tokenize(utterance))
    words = Counter(words)
    print(f'[!] whole vocab size: {len(words)}')
    words = words.most_common(cutoff)
    
    # special words
    words.extend([('<sos>', 1), ('<eos>', 1), ('<unk>', 1), 
                  ('<pad>', 1), ('<silence>', 1), ('<0>', 1), ('<1>', 1)])
    w2idx = {item[0]:idx for idx, item in enumerate(words)}
    idx2w = [item[0] for item in words]
    
    with open(vocab, 'wb') as f:
        pickle.dump((w2idx, idx2w), f)

    print(f'[!] save the vocab into {vocab}, vocab size: {len(w2idx)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='utils functions')
    parser.add_argument('--file', type=str, nargs='+', default=None, 
            help='file for generating the vocab')
    parser.add_argument('--vocab', type=str, default='',
            help='input or output vocabulary')
    parser.add_argument('--cutoff', type=int, default=50000,
            help='cutoff of the vocabulary')
    args = parser.parse_args()
    
    generate_vocab(args.file, args.vocab, cutoff=args.cutoff)
