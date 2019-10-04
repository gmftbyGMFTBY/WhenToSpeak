#!/usr/bin/python
# Author: GMFTBY
# Time: 2019.9.25

'''
utils functions for training the model or loading the dataset
'''


import pickle
import argparse
from collections import Counter
import os
import numpy as np
import torch
import nltk
from bert_serving.client import BertClient
from tqdm import tqdm
import ipdb


def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def cos_similarity(gr, ge):
    return np.dot(gr, ge) / (np.linalg.norm(gr) * np.linalg.norm(ge))


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


def load_best_model(dataset, model, net, threshold):
    path = f'./ckpt/{dataset}/{model}/'
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


def create_the_graph(turns, bc, weights=[1, 0.5], threshold=0.75, bidir=True):
    '''create the weighted directed graph of one conversation
    sequenutial edge, user connected edge, [BERT/PMI] edge
    param: turns: [turns(user, utterance)]
    param: weights: [sequential_w, user_w]
    param: bc bert encoder
    output: [2, num_edges], [num_edges]'''
    edges = {}
    s_w, u_w = weights
    # sequential edges, (turn_len - 1)
    turn_len = len(turns)
    se, ue, pe = 0, 0, 0
    for i in range(turn_len - 1):
        edges[(i, i + 1)] = [s_w]
        se += 1

    # add the self loop edge for each node
    for i in range(turn_len):
        edges[(i, i)] = [1]

    # user connected edges
    for i in range(turn_len):
        for j in range(turn_len):
            if j > i:
                useri, _ = turns[i]
                userj, _ = turns[j]
                if useri == userj:
                    if edges.get((i, j), None):
                        edges[(i, j)].append(u_w)
                    else:
                        edges[(i, j)] = [u_w]
                    ue += 1

    # PMI edges, weight by the PMI, fast
    # BERT edges, correlation is measured by the BERT embedding, slow
    utterances = []
    for user, utterance in turns:
        utterance = utterance.replace('<0>', '').strip()
        utterance = utterance.replace('<1>', '').strip()
        utterances.append(utterance)
    utterances = bc.encode(utterances)    # [turn_len, 768]

    for i in range(turn_len):
        for j in range(turn_len):
            if j > i:
                utter1, utter2 = utterances[i], utterances[j]
                weight = cos_similarity(utter1, utter2)
                if weight >= threshold:
                    if edges.get((i, j), None):
                        edges[(i, j)].append(weight)
                    else:
                        edges[(i, j)] = [weight]
                    pe += 1

    # clean the edges
    e, w = [[], []], []
    for src, tgt in edges.keys():
        e[0].append(src)
        e[1].append(tgt)
        w.append(max(edges[(src, tgt)]))

        if bidir:
            e[0].append(tgt)
            e[1].append(src)
            w.append(max(edges[(src, tgt)]))

    return (e, w), se, ue, pe


def generate_graph(dialogs, path, threshold=0.75, bidir=True):
    # dialogs: [datasize, turns]
    # return: [datasize, (2, num_edges)/ (num_edges)]
    # **make sure the bert-as-service is running**
    bc = BertClient()
    edges = []
    se, ue, pe = 0, 0, 0
    for dialog in tqdm(dialogs):
        edge, ses, ueu, pep = create_the_graph(dialog, bc, threshold=threshold, bidir=bidir)
        se += ses
        ue += ueu
        pe += pep
        edges.append(edge)

    with open(path, 'wb') as f:
        pickle.dump(edges, f)

    print(f'[!] graph information is converted in {path}')
    print(f'[!] Avg se: {round(se / len(dialogs), 4)}; Avg ue: {round(ue / len(dialogs), 4)}; Avg pe: {round(pe / len(dialogs), 4)}')

        
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


def idx2sent(data, users, vocab):
    # turn the index to the sentence
    # data: [datasize, turn, length]
    # user: [datasize, turn]
    # return: [datasize, (user, turns)]
    _, idx2w = load_pickle(vocab)
    datasets = []
    for example, user in tqdm(zip(data, users)):
        # example: [turn, length], user: [turn]
        turns = []
        for turn, u in zip(example, user):
            utterance = ' '.join([idx2w[w] for w in turn])
            utterance = utterance.replace('<1>', '').replace('<0>', '').replace('<sos>', '').replace('<eos>', '').strip()
            turns.append((u, utterance))
        datasets.append(turns)
    return datasets



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='utils functions')
    parser.add_argument('--mode', type=str, default='vocab')
    parser.add_argument('--file', type=str, nargs='+', default=None, 
            help='file for generating the vocab')
    parser.add_argument('--vocab', type=str, default='',
            help='input or output vocabulary')
    parser.add_argument('--graph', type=str, default=None)
    parser.add_argument('--cutoff', type=int, default=50000,
            help='cutoff of the vocabulary')
    parser.add_argument('--threshold', type=float, default=0.75,
            help='threshold for measuring the similarity between the utterances')
    parser.add_argument('--maxlen', type=int, default=50)
    parser.add_argument('--src_vocab', type=str, default=None)
    parser.add_argument('--tgt_vocab', type=str, default=None)
    parser.add_argument('--src', type=str, default=None)
    parser.add_argument('--tgt', type=str, default=None)
    parser.add_argument('--bidir', dest='bidir', action='store_true')
    parser.add_argument('--no-bidir', dest='bidir', action='store_false')
    args = parser.parse_args()

    if args.mode == 'vocab':
        generate_vocab(args.file, args.vocab, cutoff=args.cutoff)
    elif args.mode == 'graph':
        # save the preprocessed data for generating graph
        src_dataset, src_user, _, _, _ = load_data_cf(args.src, args.tgt, args.src_vocab, args.tgt_vocab, args.maxlen)
        print(f'[!] load the cf mode dataset, prepare for preprocessing')
        ppdataset = idx2sent(src_dataset, src_user, args.src_vocab)
        print(f'[!] begin to create the graph')
        generate_graph(ppdataset, args.graph, threshold=args.threshold, bidir=args.bidir)
