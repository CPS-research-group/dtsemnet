#!/bin/bash


# --lin_control \
# 0.005 for hard and 0.01 for easy

python -u train/train.py \
    --env_name walker \
    --policy_type drnet \
    --train_name $2 \
    --seed $1 \
    --num_leaves $3 \
    --lr $4 \
    --ddt_lr $4 \
    --buffer_size 10000000 \
    --batch_size $5 \
    --gamma 0.98 \
    --tau 0.02 \
    --learning_starts 10000 \
    --eval_freq 20000 \
    --min_reward 200 \
    --training_steps 700000 \
    --log_interval 50 \
    --lin_control \
    --use_individual_alpha \
    --train_freq 64 
  

