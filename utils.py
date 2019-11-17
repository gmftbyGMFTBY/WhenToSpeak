#!/usr/bin/python
# Author: GMFTBY
# Time: 2019.9.25

'''
1. utils functions for training the model or loading the dataset
2. the stat function for the graph
'''


import pickle
import argparse
from collections import Counter
import os
import numpy as np
import torch
import nltk
from tqdm import tqdm
import ipdb
import heapq
import random


def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def load_word_embedding(path, dimension=300):
    # load chinese or english word embedding
    unk = np.random.rand(dimension)
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    dic = {}
    for line in tqdm(lines):
        dic[line.split()[0]] = np.array([float(f) for f in line.strip().split()[1:]], dtype=np.float)
    dic['<unk>'] = unk
    return dic


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


def load_best_model(dataset, model, net, min_threshold, max_threshold):
    path = f'./ckpt/{dataset}/{model}/'
    best_loss, best_file, best_epoch = np.inf, None, -1

    for file in os.listdir(path):
        _, val_loss, _, epoch = file.split('_')
        epoch = epoch.split('.')[0]
        val_loss, epoch = float(val_loss), int(epoch)

        if min_threshold <= epoch <= max_threshold and epoch > best_epoch:
            best_file = file
            best_epoch = epoch

    if best_file:
        file_path = path + best_file
        print(f'[!] Load the model from {file_path}, threshold ({min_threshold}, {max_threshold})')
        net.load_state_dict(torch.load(file_path)['net'])
    else:
        raise Exception('[!] No saved model')


def create_the_graph(turns, weights=[1, 1], threshold=4, bidir=True):
    '''create the weighted directed graph of one conversation
    sequenutial edge, user connected edge, [BERT/PMI] edge
    param: turns: [turns(user, utterance)]
    param: weights: [sequential_w, user_w]
    param: bc bert encoder
    output: [2, num_edges], [num_edges]'''
    edges = {}
    s_w, u_w = weights
    # temporal information
    turn_len = len(turns)
    se, ue, pe = 0, 0, 0
    # for i in range(turn_len - 1):
    #     edges[(i, i + 1)] = [s_w]
    #     se += 1

    # role information
    counter = {i:1 for i in range(1, turn_len)}    # add the counter
    for i in range(turn_len):
        for j in range(turn_len):
            if j > i:
                useri, _ = turns[i]
                userj, _ = turns[j]
                # u_w = min(max(0.5 - 1 / (j - i), 0.1), 0.5)
                # add the counter
                if useri == userj:
                    if edges.get((i, j), None):
                        edges[(i, j)].append(u_w)
                    else:
                        edges[(i, j)] = [u_w]
                #     # add the counter
                #     counter[j] += 1
                    ue += 1
                
                # absolute graph
                # if edges.get((i, j), None):
                #     edges[(i, j)].append(u_w)
                # else:
                #     edges[(i, j)] = [u_w]
                # ue += 1
                
                # sparse graph
                pass

    # clean the edges
    e, w = [[], []], []
    for src, tgt in edges.keys():
        e[0].append(src)
        e[1].append(tgt)
        w.append(max(edges[(src, tgt)]))

        if bidir and src != tgt:
            # be careful of the self loop
            e[0].append(tgt)
            e[1].append(src)
            w.append(max(edges[(src, tgt)]))
            
    # if the in degree is bigger than threshold, 
    # delete (l - threshold) closest edges

    return (e, w), se, ue, pe


def generate_graph(dialogs, path, threshold=4, bidir=True):
    # dialogs: [datasize, turns]
    # return: [datasize, (2, num_edges)/ (num_edges)]
    # **make sure the bert-as-service is running**
    # bc = BertClient()
    edges = []
    se, ue, pe = 0, 0, 0
    for dialog in tqdm(dialogs):
        edge, ses, ueu, pep = create_the_graph(dialog, threshold=threshold, bidir=bidir)
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
    for sexample, texample in tqdm(zip(src, tgt)):
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


# ========== stst of the graph ========== #
def analyse_graph(path, hops=3):
    '''
    This function analyzes the graph coverage stat of the graph in Dailydialog 
    and cornell dataset.
    Stat the context node coverage of each node in the conversation.
    :param: path, the path of the dataset graph file.
    '''
    def coverage(nodes, edges):
        # return the coverage information of each node
        # return list of tuple (context nodes, coverage nodes)
        # edges to dict
        e = {}
        for i, j in zip(edges[0], edges[1]):
            if i > j:
                continue
            if e.get(j, None):
                e[j].append(i)
            else:
                e[j] = [i]
        for key in e.keys():    # make the set
            e[key] = list(set(e[key]))
        collector = []
        for node in nodes:
            # context nodes
            context_nodes = list(range(0, node))
            if context_nodes:
                # ipdb.set_trace()
                # coverage nodes, BFS
                coverage_nodes, tools, tidx = [], [(node, 0)], 0
                while True:
                    try:
                        n, nidx = tools[tidx]
                    except:
                        break
                    if nidx < hops and e.get(n, None):
                        for src in e[n]:
                            if src not in tools:
                                tools.append((src, nidx + 1))
                            if src not in coverage_nodes:
                                coverage_nodes.append(src)
                    tidx += 1
                collector.append((len(coverage_nodes), len(context_nodes)))
        return collector
    
    
    def avg_degree(ipt, opt):
        nodes = {}
        for ind, outd in zip(ipt, opt):
            if ind not in nodes:
                nodes[ind] = [0, 1]
            else:
                nodes[ind][1] += 1
            if outd not in nodes:
                nodes[outd] = [1, 0]
            else:
                nodes[outd][0] += 1
        if nodes:
            ind, outd = [i for i, j in nodes.items()], [j for i, j in nodes.items()]
            return np.mean(ind), np.mean(outd)
        else:
            return None, None
        
    graph = load_pickle(path)    # [datasize, ([2, num_edge], [num_edge])]
    avg_cover, avg_ind, avg_outd = [], [], []
    
    # degree analyse
    for idx, (edges, _) in enumerate(tqdm(graph)):
        a, b = avg_degree(*edges)
        if a:
            avg_ind.append(a)
        if b:
            avg_outd.append(b)
        
    print(f'[!] the avg in degree: {round(np.mean(avg_ind), 4)}')
    print(f'[!] the avg out degree: {round(np.mean(avg_outd), 4)}')
    
    # coverage
    avg_nodes, avg_edges = [], []
    for idx, (edges, _) in enumerate(tqdm(graph)):
        # make sure the number of the nodes
        max_n = max(edges[1]) + 1 if edges[1] else 1
        nodes = list(range(max_n))
        if max_n > 1:
            avg_nodes.append(max_n)
        if len(edges) > 1:
            avg_edges.append(len(edges[0]))
        avg_cover.extend(coverage(nodes, edges))
        
    # ========== stat ========== #
    ratio = [i / j for i, j in avg_cover]
    print(f'[!] the avg turn length of the context is {round(np.mean(avg_nodes), 4)}')
    print(f'[!] the avg edges of the context is {round(np.mean(avg_edges), 4)}')
    print(f'[!] the avg graph coverage of the context is {round(np.mean(ratio), 4)}')



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
    parser.add_argument('--threshold', type=float, default=4,
            help='threshold for measuring the similarity between the utterances')
    parser.add_argument('--maxlen', type=int, default=50)
    parser.add_argument('--src_vocab', type=str, default=None)
    parser.add_argument('--tgt_vocab', type=str, default=None)
    parser.add_argument('--src', type=str, default=None)
    parser.add_argument('--tgt', type=str, default=None)
    parser.add_argument('--hops', type=int, default=3)
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
    elif args.mode == 'stat':
        analyse_graph(args.graph, hops=args.hops)
    else:
        raise Exception('[!] wrong mode for running the utils script')
