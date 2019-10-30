#!/bin/bash
# Author: GMFTBY
# Time: 2019.9.25

mode=$1     # vocab, graph, stat, train, translate, eval
dataset=$2
model=$3
cuda=$4

epoch=30

# dataset for plus
if [ $dataset = 'cornell' ]; then
    plus=0
else
    plus=0
fi

# hierarchical
if [ $model = 'seq2seq' ]; then
    hierarchical=0
    cf=0
    graph=0
elif [ $model = 'hred' ]; then
    hierarchical=1
    cf=0
    graph=0
elif [ $model = 'hred-cf' ]; then
    hierarchical=1
    cf=1
    graph=0
elif [ $model = 'when2talk' ]; then
    hierarchical=1
    cf=1
    graph=1
elif [ $model = 'W2T_RNN_First' ]; then
    hierarchical=1
    cf=1
    graph=1
elif [ $model = 'GCNRNN' ]; then
    hierarchical=1
    cf=1
    graph=1
elif [ $model = 'W2T_GCNRNN' ]; then
    hierarchical=1
    cf=1
    graph=1
elif [ $model = 'GatedGCN' ]; then
    hierarchical=1
    cf=1
    graph=1
elif [ $model = 'GatedGCN_nobi' ]; then
    hierarchical=1
    cf=1
    graph=1
elif [ $model = 'GATRNN' ]; then
    hierarchical=1
    cf=1
    graph=1
else
    hierarchical=1
    cf=1
    graph=0
fi

# batch_size of hierarchical
if [ $hierarchical = 1 ]; then
    batch_size=64
    maxlen=50
else
    if [ $model = 'seq2seq' ] && [ $mode = 'translate' ]; then
        batch_size=64
        maxlen=100
    else
        batch_size=64
        maxlen=50
    fi
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
        --mode vocab \
        --file ./data/${dataset}-corpus/cf/src-train.pkl ./data/${dataset}-corpus/cf/src-dev.pkl \
        --vocab ./processed/$dataset/iptvocab.pkl \
        --cutoff 50000

    # generate the tgt vocab
    python utils.py \
        --mode vocab \
        --file ./data/${dataset}-corpus/cf/tgt-train.pkl ./data/${dataset}-corpus/cf/tgt-dev.pkl \
        --vocab ./processed/$dataset/optvocab.pkl \
        --cutoff 50000
        
elif [ $mode = 'stat' ]; then
    # analyse the graph information in the dataset
    echo "[!] analyze the graph coverage information"
    python utils.py \
         --mode stat \
         --graph ./processed/$dataset/train-graph.pkl \
         --hops 3
         
    python utils.py \
         --mode stat \
         --graph ./processed/$dataset/test-graph.pkl \
         --hops 3
         
    python utils.py \
         --mode stat \
         --graph ./processed/$dataset/dev-graph.pkl \
         --hops 3

elif [ $mode = 'graph' ]; then
    # generate the graph for the when2talk model
    echo "[!] create the graph of the $dataset dataset"
    python utils.py \
         --mode graph \
         --src_vocab ./processed/$dataset/iptvocab.pkl \
         --tgt_vocab ./processed/$dataset/optvocab.pkl \
         --maxlen $maxlen \
         --src ./data/${dataset}-corpus/cf/src-train.pkl \
         --tgt ./data/${dataset}-corpus/cf/tgt-train.pkl \
         --graph ./processed/$dataset/train-graph-sp.pkl \
         --threshold 4 \
         --no-bidir

    python utils.py \
        --mode graph \
        --src_vocab ./processed/$dataset/iptvocab.pkl \
        --tgt_vocab ./processed/$dataset/optvocab.pkl \
        --maxlen $maxlen \
        --src ./data/${dataset}-corpus/cf/src-test.pkl \
        --tgt ./data/${dataset}-corpus/cf/tgt-test.pkl \
        --graph ./processed/$dataset/test-graph-sp.pkl \
        --threshold 4 \
        --no-bidir

    python utils.py \
        --mode graph \
        --src_vocab ./processed/$dataset/iptvocab.pkl \
        --tgt_vocab ./processed/$dataset/optvocab.pkl \
        --maxlen $maxlen \
        --src ./data/${dataset}-corpus/cf/src-dev.pkl \
        --tgt ./data/${dataset}-corpus/cf/tgt-dev.pkl \
        --graph ./processed/$dataset/dev-graph-sp.pkl \
        --threshold 4 \
        --no-bidir

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
        --train_graph ./processed/$dataset/train-graph.pkl \
        --test_graph ./processed/$dataset/test-graph.pkl \
        --dev_graph ./processed/$dataset/dev-graph.pkl \
        --min_threshold 0 \
        --max_threshold $epoch \
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
        --epochs $epoch \
        --src_vocab ./processed/$dataset/iptvocab.pkl \
        --tgt_vocab ./processed/$dataset/optvocab.pkl \
        --maxlen $maxlen \
        --dropout 0.3 \
        --hierarchical $hierarchical \
        --cf $cf \
        --user_embed_size 10 \
        --dataset $dataset \
        --position_embed_size 30 \
        --graph $graph \
        --plus $plus \
        --contextrnn \
        --context_threshold 2

elif [ $mode = 'translate' ]; then
    CUDA_VISIBLE_DEVICES="$cuda" python translate.py \
        --src_test ./data/${dataset}-corpus/$cf_check/src-test.pkl \
        --tgt_test ./data/${dataset}-corpus/$cf_check/tgt-test.pkl \
        --test_graph ./processed/$dataset/test-graph.pkl \
        --min_threshold 0 \
        --max_threshold $epoch \
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
        --position_embed_size 30 \
        --graph $graph \
        --plus $plus \
        --contextrnn \
        --context_threshold 2

elif [ $mode = 'eval' ]; then
    CUDA_VISIBLE_DEVICES="$cuda" python eval.py \
        --model $model \
        --dataset $dataset \
        --file ./processed/$dataset/$model/pred.txt \
        --cf $cf \
        
elif [ $mode = 'curve' ]; then
    # do not add the BERTScore evaluate when begin to curve mode
    # evaluation will be too slow
    
    rm ./processed/${dataset}/${model}/conclusion.txt
    echo "[!] delete the cache file (conclusion.txt)"
    
    for i in $(seq 1 $epoch)
    do
        CUDA_VISIBLE_DEVICES="$cuda" python translate.py \
            --src_test ./data/${dataset}-corpus/$cf_check/src-test.pkl \
            --tgt_test ./data/${dataset}-corpus/$cf_check/tgt-test.pkl \
            --test_graph ./processed/$dataset/test-graph.pkl \
            --min_threshold $i \
            --max_threshold $i \
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
            --position_embed_size 30 \
            --graph $graph \
            --plus $plus \
            --contextrnn \
            --context_threshold 2

        # eval
        echo "========== $i eval ==========" >> ./processed/${dataset}/${model}/conclusion.txt
        CUDA_VISIBLE_DEVICES="$CUDA" python eval.py \
            --model $model \
            --cf $cf \
            --dataset $dataset \
            --file ./processed/${dataset}/${model}/pred.txt >> ./processed/${dataset}/${model}/conclusion.txt
            
    done


else
    echo "[!] Wrong mode for running the script"
    
fi

echo "========== $mode done =========="