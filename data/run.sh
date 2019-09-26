#!/bin/bash
# Author: GMFTBY
# Time: 2019.9.26

cf=$1    # string: cf or ncf

if [ $cf = 'cf' ]; then
    cf_check=1
elif [ $cf = 'ncf' ]; then
    cf_check=0
else
    echo "[!] wrong model (cf / ncf)"
fi

python process_ubuntu.py \
    --low 5 \
    --high 15 \
    --src_train seq2seq/$cf/src-train.pkl \
    --tgt_train seq2seq/$cf/tgt-train.pkl \
    --src_test seq2seq/$cf/src-test.pkl \
    --tgt_test seq2seq/$cf/tgt-test.pkl \
    --src_dev seq2seq/$cf/src-dev.pkl \
    --tgt_dev seq2seq/$cf/tgt-dev.pkl \
    --cf $cf_check
