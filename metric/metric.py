from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.collocations import BigramCollocationFinder
from nltk.probability import FreqDist
import argparse
import codecs
import numpy as np
import math


def cal_BLEU(refer, candidate, ngram=1):
    smoothie = SmoothingFunction().method2
    if ngram == 1:
        weight = (1, 0, 0, 0)
    elif ngram == 2:
        weight = (0.5, 0.5, 0, 0)
    elif ngram == 3:
        weight = (0.33, 0.33, 0.33, 0)
    elif ngram == 4:
        weight = (0.25, 0.25, 0.25, 0.25)
    return sentence_bleu(refer, candidate, 
                         weights=weight, 
                         smoothing_function=smoothie)


def cal_Distinct(corpus):
    """
    Calculates unigram and bigram diversity
    Args:
        corpus: tokenized list of sentences sampled
    Returns:
        uni_diversity: distinct-1 score
        bi_diversity: distinct-2 score
    """
    bigram_finder = BigramCollocationFinder.from_words(corpus)
    bi_diversity = len(bigram_finder.ngram_fd) / bigram_finder.N

    dist = FreqDist(corpus)
    uni_diversity = len(dist) / len(corpus)

    return uni_diversity, bi_diversity


def cal_BERTScore(refer, candidate):
    _, _, bert_scores = score(candidate, refer, 
                              bert="bert-base-uncased", no_idf=True)
    bert_scores = bert_scores.tolist()
    bert_scores = [0.5 if math.isnan(score) else score for score in bert_scores]
    return np.mean(bert_scores)


def cal_acc_f1(tp, fn, fp, tn):
    # return (macro-f1, micro-f1, Acc)
    acc = (tp + tn) / (tp + fn + fp + tn)
    precision_p, precision_n = tp / (tp + fp), tn / (tn + fn)
    recall_p, recall_n = tp / (tp + fn), tn / (tn + fp)
    avg_pre, avg_recall = (precision_n + precision_p) / 2, (recall_p + recall_n) / 2
    macro_f1 = 2 * avg_pre * avg_recall / (avg_pre + avg_recall)
    mi_pre = (tp + tn) / (tp + fp + tn + fn)
    mi_rec = (tp + tn) / (tp + fn + tn + fp)
    micro_f1 = 2 * mi_pre * mi_rec / (mi_pre + mi_rec)
    return macro_f1, micro_f1, acc


def cal_acc_P_R_F1(tp, fn, fp, tn):
    # cal the F1 metric from the stat data of the postive label
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (tp + tn) / (tp + fn + fp + tn)
    return round(precision, 4), round(recall, 4), round(f1, 4), round(acc, 4)


if __name__ == "__main__":
    pass
