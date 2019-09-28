#!/bin/bash
# Author: GMFTBY
# Time: 2019.9.25

mode=$1     # vocab, train, translate, eval
dataset=$2
model=$3
cuda=$4

# hierarchical
if [ $model = 'seq2seq' ]; then
    hierarchical=0
    cf=0
elif [ $model = 'hred' ]; then
    hierarchical=1
    cf=0
elif [ $model = 'hred-cf' ]; then
    hierarchical=1
    cf=1
else
    hierarchical=1
    cf=1
fi

# batch_size of hierarchical
if [ $hierarchical = 1 ]; then
    batch_size=32
    maxlen=50
else
    batch_size=32
    maxlen=100
fi

# cf_check
if [ $cf = 1 ]; then
    cf_check="cf"
else
    cf_check="ncf"
fi

echo "========== $mode begin =========="


if [ $mode = 'vocab' ]; then
    # generate the src vocab
    python utils.py \
        --file ./data/${dataset}-corpus/cf/src-train.pkl ./data/${dataset}-corpus/cf/src-dev.pkl \
        --vocab ./processed/$dataset/iptvocab.pkl \
        --cutoff 50000

    # generate the tgt vocab
    python utils.py \
        --file ./data/${dataset}-corpus/cf/tgt-train.pkl ./data/${dataset}-corpus/cf/tgt-dev.pkl \
        --vocab ./processed/$dataset/optvocab.pkl \
        --cutoff 50000

elif [ $mode = 'train' ]; then
    # clear the ckpt, vocab, tensorboard cache
    rm ./ckpt/$dataset/$model/*
    rm ./tblogs/$dataset/$model/*
    
    # train loop
    CUDA_VISIBLE_DEVICES="$cuda" python train.py \
        --src_train ./data/${dataset}-corpus/$cf_check/src-train.pkl \
        --tgt_train ./data/${dataset}-corpus/$cf_check/tgt-train.pkl \
        --src_test ./data/${dataset}-corpus/$cf_check/src-test.pkl \
        --tgt_test ./data/${dataset}-corpus/$cf_check/tgt-test.pkl \
        --src_dev ./data/${dataset}-corpus/$cf_check/src-dev.pkl \
        --tgt_dev ./data/${dataset}-corpus/$cf_check/tgt-dev.pkl \
        --epoch_threshold 0 \
        --lr 1e-4 \
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
        --src_vocab ./processed/$dataset/iptvocab.pkl \
        --tgt_vocab ./processed/$dataset/optvocab.pkl \
        --maxlen $maxlen \
        --dropout 0.3 \
        --hierarchical $hierarchical \
        --cf $cf \
        --user_embed_size 10 \
        --dataset $dataset \

elif [ $mode = 'translate' ]; then
    CUDA_VISIBLE_DEVICES="$cuda" python translate.py \
        --src_test ./data/${dataset}-corpus/$cf_check/src-test.pkl \
        --tgt_test ./data/${dataset}-corpus/$cf_check/tgt-test.pkl \
        --epoch_threshold 0 \
        --batch_size $batch_size \
        --model $model \
        --utter_n_layer 2 \
        --utter_hidden 500 \
        --context_hidden 500 \
        --decoder_hidden 500 \
        --embed_size 300 \
        --seed 20 \
        --src_vocab ./processed/$dataset/iptvocab.pkl \
        --tgt_vocab ./processed/$dataset/optvocab.pkl \
        --maxlen $maxlen \
        --pred ./processed/$dataset/$model/pred.txt \
        --hierarchical $hierarchical \
        --tgt_maxlen 50 \
        --cf $cf \
        --user_embed_size 10 \
        --dataset $dataset \

elif [ $mode = 'eval' ]; then
    CUDA_VISIBLE_DEVICES="$cuda" python eval.py \
        --model $model \
        --dataset $dataset \
        --file ./processed/$dataset/$model/pred.txt \
        --cf $cf \

else
    echo "[!] Wrong mode for running the script"
fi

echo "========== $mode done =========="
