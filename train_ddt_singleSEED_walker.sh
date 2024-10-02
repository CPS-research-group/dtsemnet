#!/bin/bash
# --submodels \

python -u train/trainH.py \
    --env_name walker \
    --policy_type ddt \
    --train_name $2 \
    --seed $1 \
    --num_leaves $3 \
    --lr $4 \
    --ddt_lr $4 \
    --buffer_size 1000000 \
    --batch_size $5 \
    --gamma 0.99 \
    --tau 0.005 \
    --learning_starts 10000 \
    --eval_freq 20000 \
    --min_reward 250 \
    --training_steps 800000 \
    --log_interval 50 \
    --use_individual_alpha \
    --hard_node \
    --submodels \
    --argmax_tau 1.0 \
    --sparse_submodel_type 0
