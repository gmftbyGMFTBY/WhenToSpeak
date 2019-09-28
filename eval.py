#!/usr/bin/python3
# Author: GMFTBY
# Time: 2019.9.19

from metric.metric import * 
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument('--model', type=str, default='HRED', help='model name')
    parser.add_argument('--file', type=str, default=None, help='result file')
    args = parser.parse_args()

    with open(args.file) as f:
        ref, tgt = [], []
        for idx, line in enumerate(f.readlines()):
            if idx % 4 == 1:
                line = line.replace("- ref: ", "").replace('<sos>', '').replace('<eos>', '').strip()
                ref.append(line.split())
            elif idx % 4 == 2:
                line = line.replace("- tgt: ", "").replace('<sos>', '').replace('<eos>', '').strip()
                tgt.append(line.split())

    assert len(ref) == len(tgt)

    # BLEU
    bleu_sum, counter = 0, 0
    for rr, cc in zip(ref, tgt):
        bleu_sum += cal_BLEU([rr], cc, ngram=4)
        counter += 1

    # Distinct-1, Distinct-2
    candidates = []
    for line in tgt:
        candidates.extend(line)
    distinct_1, distinct_2 = cal_Distinct(candidates)

    # BERTScore < 512 for bert
    ref = [' '.join(i)[:512] for i in ref]
    tgt = [' '.join(i)[:512] for i in tgt]
    bert_scores = cal_BERTScore(ref, tgt)

    print(f'Model {args.model} Result')
    print(f'BLEU-4: {round(bleu_sum / counter, 4)}')
    print(f'Distinct-1: {round(distinct_1, 4)}; Distinct-2: {round(distinct_2, 4)}')
    print(f'BERTScore: {round(bert_scores, 4)}')
