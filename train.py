#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.9.15

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import random
import numpy as np
import argparse
import math
import pickle
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import ipdb

from utils import *
from data_loader import *
from model.seq2seq_attention import Seq2Seq
from model.HRED import HRED
from model.seq2seq_attention_cf import Seq2Seq_cf
from model.layers import *


def train(writer, writer_str, train_iter, net, optimizer, vocab_size, pad, 
          grad_clip=10, cf=False):
    # choose nll_loss for training the objective function
    net.train()
    total_loss, batch_num = 0.0, 0
    criterion = nn.NLLLoss(ignore_index=pad)
    de_criterion = nn.BCELoss()

    pbar = tqdm(train_iter)

    for idx, batch in enumerate(pbar):
        # [turn, length, batch], [seq_len, batch] / [seq_len, batch], [seq_len, batch]
        if cf:
            # debatch: [batch,]
            sbatch, tbatch, debatch, turn_lengths = batch
        else:
            sbatch, tbatch, turn_lengths = batch
        
        batch_size = tbatch.shape[1]
        if batch_size == 1:
            # batchnorm will throw error when batch_size is 1
            continue

        optimizer.zero_grad()

        # [seq_len, batch, vocab_size]
        output = net(sbatch, tbatch, turn_lengths)
        if cf:
            output, de = output
            de_loss = de_criterion(de, debatch)
            lm_loss = criterion(output[1:].view(-1, vocab_size),
                                tbatch[1:].contiguous().view(-1))
            loss = 0.75 * de_loss + 0.25 * lm_loss
            writer.add_scalar(f'{writer_str}-DeLoss/train', de_loss, idx)
            writer.add_scalar(f'{writer_str}-LMLoss/train', lm_loss, idx)
            writer.add_scalar(f'{writer_str}-Loss/train', loss, idx)
        else:
            loss = criterion(output[1:].view(-1, vocab_size),
                             tbatch[1:].contiguous().view(-1))
            # add train loss to the tensorfboard
            writer.add_scalar(f'{writer_str}-Loss/train', loss, idx)

        loss.backward()
        clip_grad_norm_(net.parameters(), grad_clip)

        optimizer.step()
        total_loss += loss.item()
        batch_num += 1

        pbar.set_description(f'batch {batch_num}, training loss: {round(loss.item(), 4)}, lr: {round(optimizer.param_groups[0]["lr"], 10)}')

    # return avg loss
    return round(total_loss / batch_num, 4)


def validation(data_iter, net, vocab_size, pad, cf=False):
    net.eval()
    batch_num, total_loss, total_acc, total_num = 0, 0.0, 0, 0
    criterion = nn.NLLLoss(ignore_index=pad)
    de_criterion = nn.BCELoss()

    pbar = tqdm(data_iter)

    for idx, batch in enumerate(pbar):
        if cf:
            sbatch, tbatch, debatch, turn_lengths = batch
        else:
            sbatch, tbatch, turn_lengths = batch
        batch_size = tbatch.shape[1]
        if batch_size == 1:
            continue

        output = net(sbatch, tbatch, turn_lengths)
        if cf:
            output, de = output
            de_loss = de_criterion(de, debatch)
            lm_loss = criterion(output[1:].view(-1, vocab_size),
                                tbatch[1:].view(-1, vocab_size))
            loss = 0.75 * de_loss + 0.25 * lm_loss
            # accuracy of the decision output
            # de: [batch]
            de = (de > 0.5).long()
            total_acc += torch.sum(de == debatch).item()
            total_num += len(debatch)
        else:
            loss = criterion(output[1:].view(-1, vocab_size),
                             tbatch[1:].contiguous().view(-1))
        pbar.set_description(f'batch {idx}, dev/test loss: {round(loss.item(), 4)}')
        total_loss += loss.item()
        batch_num += 1

    if cf:
        return round(total_loss / batch_num, 4), round(total_acc / total_num, 4)
    else:
        return round(total_loss / batch_num, 4)


def test(data_iter, net, vocab_size, pad, cf=False):
    test_loss = validation(data_iter, net, vocab_size, pad, cf=cf)
    return test_loss


def main(**kwargs):
    # tensorboard 
    writer = SummaryWriter(log_dir=f'./tblogs/{kwargs["model"]}')

    # load vocab
    src_vocab, tgt_vocab = load_pickle(kwargs['src_vocab']), load_pickle(kwargs['tgt_vocab'])
    src_w2idx, src_idx2w = src_vocab
    tgt_w2idx, tgt_idx2w = tgt_vocab

    # create the net
    if kwargs['model'] == 'seq2seq':
        net = Seq2Seq(len(src_w2idx), kwargs['embed_size'], len(tgt_w2idx), 
                      kwargs['utter_hidden' ], 
                      kwargs['decoder_hidden'], teach_force=kwargs['teach_force'],
                      pad=tgt_w2idx['<pad>'], sos=tgt_w2idx['<sos>'],
                      dropout=kwargs['dropout'], 
                      utter_n_layer=kwargs['utter_n_layer'])
    elif kwargs['model'] == 'hred':
        net = HRED(kwargs['embed_size'], len(src_w2idx), len(tgt_w2idx), 
                   kwargs['utter_hidden'], kwargs['context_hidden'], 
                   kwargs['decoder_hidden'], teach_force=kwargs['teach_force'],
                   pad=tgt_w2idx['<pad>'], sos=tgt_w2idx['<sos>'],
                   dropout=kwargs['dropout'], utter_n_layer=kwargs['utter_n_layer'])
    elif kwargs['model'] == 'seq2seq-cf':
        net = Seq2Seq_cf(len(src_w2idx), kwargs['embed_size'], len(tgt_w2idx),
                         kwargs['utter_hidden'], kwargs['decoder_hidden'], 
                         teach_force=kwargs['teach_force'], pad=tgt_w2idx['<pad>'],
                         sos=tgt_w2idx['<sos>'], dropout=kwargs['dropout'],
                         utter_n_layer=kwargs['utter_n_layer'])
    elif kwargs['model'] == 'hred-cf':
        pass
    else:
        raise Exception('[!] Wrong model (seq2seq, hred, seq2seq-cf, hred-cf)')

    if torch.cuda.is_available():
        net.cuda()

    print('[!] Net:')
    print(net)
    print(f'[!] Parameters size: {sum(x.numel() for x in net.parameters())}')
    print(f'[!] Optimizer Adam')
    optimizer = optim.Adam(net.parameters(), lr=kwargs['lr'], 
                           weight_decay=kwargs['weight_decay'])

    pbar = tqdm(range(1, kwargs['epochs'] + 1))
    training_loss, validation_loss = [], []
    min_loss = np.inf
    patience = 0
    best_val_loss = None

    # train
    for epoch in pbar:
        # prepare dataset
        if kwargs['hierarchical'] == 1:
            train_iter = get_batch_data(kwargs['src_train'], kwargs['tgt_train'],
                                        kwargs['src_vocab'], kwargs['tgt_vocab'], 
                                        kwargs['batch_size'], kwargs['maxlen'])
            test_iter = get_batch_data(kwargs['src_test'], kwargs['tgt_test'],
                                       kwargs['src_vocab'], kwargs['tgt_vocab'],
                                       kwargs['batch_size'], kwargs['maxlen'])
            dev_iter = get_batch_data(kwargs['src_dev'], kwargs['tgt_dev'],
                                      kwargs['src_vocab'], kwargs['tgt_vocab'],
                                      kwargs['batch_size'], kwargs['maxlen'])
        else:
            train_iter = get_batch_data_flatten(kwargs['src_train'], kwargs['tgt_train'],
                                                kwargs['src_vocab'], kwargs['tgt_vocab'],
                                                kwargs['batch_size'], kwargs['maxlen'])
            test_iter = get_batch_data_flatten(kwargs['src_test'], kwargs['tgt_test'],
                                               kwargs['src_vocab'], kwargs['tgt_vocab'],
                                               kwargs['batch_size'], kwargs['maxlen'])
            dev_iter = get_batch_data_flatten(kwargs['src_dev'], kwargs['tgt_dev'],
                                              kwargs['src_vocab'], kwargs['tgt_vocab'],
                                              kwargs['batch_size'], kwargs['maxlen'])

        writer_str = f'{kwargs["model"]}-epoch-{epoch}'
        train(writer, writer_str, train_iter, net, optimizer, 
              len(tgt_w2idx), tgt_w2idx['<pad>'], 
              grad_clip=kwargs['grad_clip'], cf=kwargs['cf']==1)
        if kwargs["cf"] == 1:
            val_loss, val_acc = validation(dev_iter, net, len(tgt_w2idx), tgt_w2idx['<pad>'], 
                                           cf=kwargs["cf"]==1)
            writer.add_scalar(f'{kwargs["model"]}-Acc/dev', val_acc, epoch)
        else:
            val_loss = validation(dev_iter, net, len(tgt_w2idx), tgt_w2idx['<pad>'], 
                                  cf=kwargs["cf"]==1)
        # add scalar to tensorboard
        writer.add_scalar(f'{kwargs["model"]}-Loss/dev', val_loss, epoch)

        if not best_val_loss or val_loss < best_val_loss:
            state = {'net': net.state_dict(), 'epoch': epoch}
            torch.save(state, 
                       f'./ckpt/{kwargs["model"]}/vloss_{val_loss}_epoch_{epoch}.pt')
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
        
        pbar.set_description(f'Epoch: {epoch}, val_loss: {val_loss}, val_ppl: {round(math.exp(val_loss), 4)}, patience: {patience}/{kwargs["patience"]}')

        if patience > kwargs['patience']:
            print(f'Early Stop {kwargs["patience"]} at epoch {epoch}')
            break

    pbar.close()

    # test
    load_best_model(kwargs['model'], net, threshold=kwargs['epoch_threshold'])
    test_loss = test(test_iter, net, len(tgt_w2idx), tgt_w2idx['<pad>'], cf=kwargs["cf"])
    print(f'Test loss: {test_loss}, test_ppl: {round(math.exp(test_loss), 4)}')
    writer.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('--src_train', type=str, default=None, help='src train file')
    parser.add_argument('--tgt_train', type=str, default=None, help='src train file')
    parser.add_argument('--src_test', type=str, default=None, help='src test file')
    parser.add_argument('--tgt_test', type=str, default=None, help='tgt test file')
    parser.add_argument('--src_dev', type=str, default=None, help='src dev file')
    parser.add_argument('--tgt_dev', type=str, default=None, help='tgt dev file')
    parser.add_argument('--epoch_threshold', type=int, default=20, 
                        help='epoch threshold for loading best model')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--model', type=str, default='HRED', help='model to be trained')
    parser.add_argument('--utter_hidden', type=int, default=150, 
                        help='utterance encoder hidden size')
    parser.add_argument('--teach_force', type=float, default=0.5, help='teach force ratio')
    parser.add_argument('--context_hidden', type=int, default=150, 
                        help='context encoder hidden size')
    parser.add_argument('--decoder_hidden', type=int, default=150, 
                        help='decoder hidden size')
    parser.add_argument('--seed', type=int, default=30,
                        help='random seed')
    parser.add_argument('--embed_size', type=int, default=200, 
                        help='embedding layer size')
    parser.add_argument('--patience', type=int, default=25, help='patience for early stop')
    parser.add_argument('--grad_clip', type=float, default=10.0, help='grad clip')
    parser.add_argument('--epochs', type=int, default=100, help='epochs for training')
    parser.add_argument('--src_vocab', type=str, default=None, help='src vocabulary')
    parser.add_argument('--tgt_vocab', type=str, default=None, help='tgt vocabulary')
    parser.add_argument('--maxlen', type=int, default=50, help='the maxlen of the utterance')
    parser.add_argument('--utter_n_layer', type=int, default=1, 
                        help='layers of the utterance encoder')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--hierarchical', type=int, default=1, help='Whether hierarchical architecture')
    parser.add_argument('--cf', type=int, default=0, help='whether have the classification')

    args = parser.parse_args()

    # show the parameters
    print('[!] Parameters:')
    print(args)

    # set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # main function
    args_dict = vars(args)
    main(**args_dict)