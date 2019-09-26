#!/bin/bash
# Author: GMFTBY
# Time: 2019.9.25

mode=$1
model=$2
cuda=$3

# hierarchical
if [ $model = 'seq2seq' ]; then
    hierarchical=0
elif [ $model = 'seq2seq-cf' ]; then
    hierarchical=0
elif [ $model = 'hred' ]; then
    hierarchical=1
elif [ $model = 'hred-cf' ]; then
    hierarchical=1
fi

# batch_size of hierarchical
if [ $hierarchical = 1 ]; then
    batch_size=32
    maxlen=50
else
    batch_size=32
    maxlen=120
fi

echo "========== $mode begin =========="

if [ $mode = 'train' ]; then
    # clear the ckpt, vocab, tensorboard cache
    rm ./ckpt/$model/*
    rm ./processed/$model/*
    rm ./tblogs/$model/*

    # generate the src vocab
    python utils.py \
        --file ./data/$model/src-train.pkl ./data/$model/src-dev.pkl \
        --vocab ./processed/$model/iptvocab.pkl \
        --cutoff 50000

    # generate the tgt vocab
    python utils.py \
        --file ./data/$model/tgt-train.pkl ./data/$model/tgt-dev.pkl \
        --vocab ./processed/$model/optvocab.pkl \
        --cutoff 50000

    # train loop
    CUDA_VISIBLE_DEVICES="$cuda" python train.py \
        --src_train ./data/$model/src-train.pkl \
        --tgt_train ./data/$model/tgt-train.pkl \
        --src_test ./data/$model/src-test.pkl \
        --tgt_test ./data/$model/tgt-test.pkl \
        --src_dev ./data/$model/src-dev.pkl \
        --tgt_dev ./data/$model/tgt-dev.pkl \
        --epoch_threshold 0 \
        --lr 1e-3 \
        --batch_size $batch_size \
        --weight_decay 1e-6 \
        --model $model \
        --teach_force 1 \
        --utter_hidden 500 \
        --context_hidden 500 \
        --decoder_hidden 500 \
        --utter_n_layer 2 \
        --seed 20 \
        --embed_size 300 \
        --patience 10 \
        --grad_clip 3 \
        --epochs 50 \
        --src_vocab ./processed/$model/iptvocab.pkl \
        --tgt_vocab ./processed/$model/optvocab.pkl \
        --maxlen $maxlen \
        --dropout 0.3 \
        --hierarchical $hierarchical \

elif [ $mode = 'translate' ]; then
    CUDA_VISIBLE_DEVICES="$cuda" python translate.py \
        --src_test ./data/$model/src-test.pkl \
        --tgt_test ./data/$model/tgt-test.pkl \
        --epoch_threshold 0 \
        --batch_size $batch_size \
        --model $model \
        --utter_n_layer 2 \
        --utter_hidden 500 \
        --context_hidden 500 \
        --decoder_hidden 500 \
        --embed_size 300 \
        --seed 20 \
        --src_vocab ./processed/$model/iptvocab.pkl \
        --tgt_vocab ./processed/$model/optvocab.pkl \
        --maxlen $maxlen \
        --pred ./processed/$model/pred.txt \
        --hierarchical $hierarchical \
        --tgt_maxlen 50 \

elif [ $mode = 'eval' ]; then
    CUDA_VISIBLE_DEVICES="$cuda" python eval.py \
        --model $model \
        --file ./processed/$model/pred.txt
else
    echo "[!] Wrong mode for running the script"
fi

echo "========== $mode Done =========="
