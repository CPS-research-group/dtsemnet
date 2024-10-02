#!/bin/bash

python -u train/train.py \
    --env_name lunar \
    --policy_type ddt \
    --train_name $2 \
    --seed $1 \
    --num_leaves $3 \
    --lr 1e-4 \
    --ddt_lr 1e-4 \
    --buffer_size 1000000 \
    --batch_size 256 \
    --gamma 0.99 \
    --tau 0.01 \
    --learning_starts 10000 \
    --eval_freq 1500 \
    --min_reward 250 \
    --training_steps 500000 \
    --log_interval 50 \
    --use_individual_alpha \
    --hard_node \
    --submodels \
    --argmax_tau 1.0 \
    --sparse_submodel_type 0

