#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.9.19

from metric.metric import * 
import argparse
import ipdb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument('--model', type=str, default='HRED', help='model name')
    parser.add_argument('--dataset', type=str, default='ubuntu')
    parser.add_argument('--file', type=str, default=None, help='result file')
    parser.add_argument('--cf', type=int, default=1, help='cf mode')
    args = parser.parse_args()

    # load the file data
    tp, fn, fp, tn = 0, 0, 0, 0
    rl, tl = False, False
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
                else:
                    ref.append(srcline.split())
                    tgt.append(tgtline.split())


    assert len(ref) == len(tgt)

    # BLEU
    bleu1_sum, bleu2_sum, bleu3_sum, bleu4_sum, counter = 0, 0, 0, 0, 0
    for rr, cc in zip(ref, tgt):
        bleu1_sum += cal_BLEU([rr], cc, ngram=1)
        bleu2_sum += cal_BLEU([rr], cc, ngram=2)
        bleu3_sum += cal_BLEU([rr], cc, ngram=3)
        bleu4_sum += cal_BLEU([rr], cc, ngram=4)
        counter += 1

    # Distinct-1, Distinct-2
    candidates = []
    for line in tgt:
        candidates.extend(line)
    # ipdb.set_trace()
    distinct_1, distinct_2 = cal_Distinct(candidates)

    print(f'Model {args.model} Result')
    print(f'BLEU-4: {round(bleu4_sum / counter, 4)}')
    print(f'Distinct-1: {round(distinct_1, 4)}; Distinct-2: {round(distinct_2, 4)}')
    # print(f'BERTScore: {round(bert_scores, 4)}')
    
    if args.cf == 1:
        macro_f1, micro_f1, acc = cal_acc_f1(tp, fn, fp, tn)
        print(f'Decision Acc: {acc}')
        print(f'Decision macro-F1: {macro_f1}, Decision micro-F1: {micro_f1}')