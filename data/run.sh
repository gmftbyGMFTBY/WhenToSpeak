#!/bin/bash
# Author: GMFTBY
# Time: 2019.9.26

dataset=$1
cf=$2       # string: cf or ncf

if [ $cf = 'cf' ]; then
    cf_check=1
elif [ $cf = 'ncf' ]; then
    cf_check=0
else
    echo "[!] wrong model (cf / ncf)"
fi

# dailydialog(5~35), cornell(5~15)
python process.py \
    --low 5 \
    --high 35 \
    --dataset $dataset \
    --maxsize 20000 \
    --src_train ${dataset}-corpus/$cf/src-train.pkl \
    --tgt_train ${dataset}-corpus/$cf/tgt-train.pkl \
    --src_test ${dataset}-corpus/$cf/src-test.pkl \
    --tgt_test ${dataset}-corpus/$cf/tgt-test.pkl \
    --src_dev ${dataset}-corpus/$cf/src-dev.pkl \
    --tgt_dev ${dataset}-corpus/$cf/tgt-dev.pkl \
    --cf $cf_check
