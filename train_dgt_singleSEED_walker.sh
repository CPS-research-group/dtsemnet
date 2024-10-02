#!/bin/bash

# lr 1e-3 0.001
python -u train/train.py \
    --env_name walker \
    --policy_type dgt \
    --train_name $2 \
    --seed $1 \
    --num_leaves $3 \
    --lr $4 \
    --ddt_lr $4 \
    --buffer_size 1000000 \
    --batch_size $5 \
    --gamma 0.98 \
    --tau 0.02 \
    --learning_starts 10000 \
    --eval_freq 1500 \
    --min_reward 250 \
    --training_steps 700000 \
    --log_interval 50 \
    --use_individual_alpha \
  

