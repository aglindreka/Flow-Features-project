#!/usr/bin/env bash
export PATH=/pytorch_env/bin:$PATH

python train.py \
-dataset charades \
-mode rgb \
-model MS_TCT \
-train True \
-num_clips 2100 \
-skip 0 \
-lr 0.001 \
-comp_info False \
-epoch 100 \
-unisize True \
-alpha_l 1 \
-beta_l 0.05 \
-batch_size 1