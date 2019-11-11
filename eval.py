#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.9.19

from metric.metric import * 
import argparse
import random
from utils import load_word_embedding
import pickle
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument('--model', type=str, default='HRED', help='model name')
    parser.add_argument('--dataset', type=str, default='ubuntu')
    parser.add_argument('--file', type=str, default=None, help='result file')
    parser.add_argument('--cf', type=int, default=1, help='cf mode')
    parser.add_argument('--embedding', type=str, default='/home/lt/data/File/wordembedding/glove/glove.6B.300d.txt')
    parser.add_argument('--dim', type=int, default=300)
    args = parser.parse_args()
    
    # create the word embedding
    # dic = load_word_embedding(args.embedding, dimension=args.dim)
    with open('./data/dict.pkl', 'rb') as f:
        dic = pickle.load(f)

    # load the file data
    tp, fn, fp, tn = 0, 0, 0, 0
    rl, tl = False, False
    silence_wrong, whole_counter = 0, 0
    with open(args.file) as f:
        ref, tgt = [], []
        for idx, line in enumerate(f.readlines()):
            if idx % 4 == 1:
                if "- ref:" in line:
                    rl = False
                elif "+ ref:" in line:
                    rl = True
                srcline = line.replace("- ref: ", "").replace('<sos>', '').replace('<eos>', '').strip()
                srcline = srcline.replace("+ ref: ", "").replace('<sos>', '').replace('<eos>', '').strip()
            elif idx % 4 == 2:
                if "- tgt:" in line:
                    tl = False
                elif "+ tgt:" in line:
                    tl = True
                tgtline = line.replace("- tgt: ", "").replace('<sos>', '').replace('<eos>', '').strip()
                tgtline = tgtline.replace("+ tgt: ", "").replace('<sos>', '').replace('<eos>', '').strip()
            elif idx % 4  == 3:
                # counter
                whole_counter += 1
                # stat the tp, fn, fp, tn
                if rl and tl:
                    tp += 1
                elif rl and not tl:
                    fn += 1
                elif not rl and tl:
                    fp += 1
                else:
                    tn += 1

                if args.cf == 1:
                    if rl and tl:
                        ref.append(srcline.split())
                        tgt.append(tgtline.split())
                    if (tl and 'silence' in tgtline) or (not tl and 'silence' not in tgtline):
                        silence_wrong += 1
                else:
                    if 'silence' in tgtline or 'silence' in srcline:
                        pass
                    else:
                        ref.append(srcline.split())
                        tgt.append(tgtline.split())
                    
    # filter
    if args.cf == 0:
        idx_ = random.sample(list(range(len(ref))), int(0.85 * len(ref)))
        ref = [i for idx, i in enumerate(ref) if idx in idx_]
        tgt = [i for idx, i in enumerate(tgt) if idx in idx_]
    else:
        print(f'[!] test ({len(ref)}|{round(len(ref) / (tp + fn), 4)}) examples')
        print(f'[!] true acc: {round(tp / (tp + fn), 4)}, false acc: {round(tn / (tn + fp), 4)}')
        print(f'[!] silence error ratio: {round(silence_wrong / whole_counter, 4)}')
        
    assert len(ref) == len(tgt)

    # BLEU and embedding-based metric
    bleu1_sum, bleu2_sum, bleu3_sum, bleu4_sum, embedding_average_sum, counter, ve_sum = 0, 0, 0, 0, 0, 0, 0
    for rr, cc in tqdm(zip(ref, tgt)):
        bleu1_sum += cal_BLEU([rr], cc, ngram=1)
        bleu2_sum += cal_BLEU([rr], cc, ngram=2)
        bleu3_sum += cal_BLEU([rr], cc, ngram=3)
        bleu4_sum += cal_BLEU([rr], cc, ngram=4)
        embedding_average_sum += cal_embedding_average(rr, cc, dic)
        ve_sum += cal_vector_extrema(rr, cc, dic)
        counter += 1

    # Distinct-1, Distinct-2
    candidates = []
    for line in tgt:
        candidates.extend(line)
    # ipdb.set_trace()
    distinct_1, distinct_2 = cal_Distinct(candidates)

    print(f'Model {args.model} Result')
    print(f'BLEU-1: {round(bleu1_sum / counter, 4)}')
    print(f'BLEU-2: {round(bleu2_sum / counter, 4)}')
    print(f'BLEU-3: {round(bleu3_sum / counter, 4)}')
    print(f'BLEU-4: {round(bleu4_sum / counter, 4)}')
    print(f'Embedding Average: {round(embedding_average_sum / counter, 4)}')
    print(f'Vector Extrema: {round(ve_sum / counter, 4)}')
    print(f'Distinct-1: {round(distinct_1, 4)}; Distinct-2: {round(distinct_2, 4)}')
    # print(f'BERTScore: {round(bert_scores, 4)}')
    
    if args.cf == 1:
        macro_f1, micro_f1, acc = cal_acc_f1(tp, fn, fp, tn)
        print(f'Decision Acc: {acc}')
        print(f'Decision macro-F1: {macro_f1}, Decision micro-F1: {micro_f1}')