#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.9.25


'''
Process the ubuntu dialogue dataset v1.0 into train test dev dataset

Before you run this script, make sure you clean the folder
'''

import csv
import os
import pickle
import copy
import argparse
import random
import ipdb
from tqdm import tqdm


def get_all_dialogues(path, turns_threshold=(5, 15)):
    # path: dataset folder filter the dialog by turns threshold
    less, more = turns_threshold
    print(f'Ignore the dialogue turns that less than {less} or more than {more}')
    files = []
    for folder in os.listdir(path):
        npath = os.path.join(path, folder)
        if int(folder) > more or int(folder) < less:
            continue
        for file in os.listdir(npath):
            fpath = os.path.join(npath, file)
            files.append(fpath)
    return files


def process_one_dialog(path, cf):
    # only return the dialog that have two users
    # cf: if 1, not combine; if 0, combine.
    with open(path) as f:
        f_csv = csv.reader(f, delimiter='\t')
        users, utterances = [], []
        last_user, cache = None, []
        for line in f_csv:
            _, u1, u2, utterance = line
            if u1 and u1 not in users: users.append(u1)
            if u2 and u2 not in users: users.append(u2)

            if cf == 0:
                if not last_user or last_user == u1:
                    last_user = u1
                    cache.append(utterance)
                else:
                    uu = " <eou> ".join(cache)
                    cache = [utterance]
                    utterances.append((last_user, uu))
                    last_user = u1
            else:
                if len(utterances) > 0 and utterance == utterances[-1][-1]:
                    # ignore the useless utterance which is bad for the model
                    pass
                else:
                    utterances.append((u1, utterance))
        if cf == 0 and cache:
            utterances.append((last_user, " <eou> ".join(cache)))

        if len(users) != 2:
            return (users, utterances)
        else:
            return None


def make_src_tgt(dialogs):
    # dialog: [(users, [(user, utterance)])], make the pickle file
    def one_dialog(dialog):
        users, utterances = dialog
        cache, src, tgt = [], [], []
        for turn in utterances:
            user, utterance = turn
            utterance = f'<{users.index(user)}> {utterance}'
    
            if cache:
                src.append(copy.deepcopy(cache))
                tgt.append([(f'<{users.index(user)}>', utterance)])
            cache.append((f'<{users.index(user)}>', utterance))
        return src, tgt

    src, tgt = [], []
    for dialog in dialogs:
        s, t = one_dialog(dialog)
        src.extend(s)
        tgt.extend(t)

    return (src, tgt)


def write_file(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'object write into {path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process the ubuntu dataset')
    parser.add_argument('--low', type=int, default=5, help='Low threshold')
    parser.add_argument('--high', type=int, default=15, help='High threshold')
    parser.add_argument('--src_train', type=str, default='seq2seq/src-train.pkl')
    parser.add_argument('--tgt_train', type=str, default='seq2seq/tgt-train.pkl')
    parser.add_argument('--src_test', type=str, default='seq2seq/src-test.pkl')
    parser.add_argument('--tgt_test', type=str, default='seq2seq/tgt-test.pkl')
    parser.add_argument('--src_dev', type=str, default='seq2seq/src-dev.pkl')
    parser.add_argument('--tgt_dev', type=str, default='seq2seq/tgt-dev.pkl')
    parser.add_argument('--cf', type=int, default=0, help='whether have the classification')
    args = parser.parse_args()

    files = get_all_dialogues('./dialogs', turns_threshold=(args.low, args.high))

    dialogs = []
    for file in tqdm(files):
        dialog = process_one_dialog(file, args.cf)
        if dialog:
            dialogs.append(dialog)
    
    print(f'Totally {len(dialogs)} dialogs')
    
    src, tgt = make_src_tgt(dialogs)
    idx = list(range(0, len(src)))
    src = [src[i] for i in idx]
    tgt = [tgt[i] for i in idx]
    print(f'Totally {len(src)} examples')

    train_size = int(len(src) * 0.95)
    test_size = int((len(src) - train_size) * 0.5)
    dev_size = len(src) - train_size - test_size

    src_train, tgt_train = src[:train_size], tgt[:train_size]
    src_test, tgt_test = src[train_size:train_size+test_size], tgt[train_size:train_size+test_size]
    src_dev, tgt_dev = src[-dev_size:], tgt[-dev_size:]

    print(f'Train size: {len(src_train)}, test size: {len(src_test)}, dev size: {len(src_dev)}')

    write_file(src_train, args.src_train)
    write_file(tgt_train, args.tgt_train)
    write_file(src_test, args.src_test)
    write_file(tgt_test, args.tgt_test)
    write_file(src_dev, args.src_dev)
    write_file(tgt_dev, args.tgt_dev)

    print('Done')
