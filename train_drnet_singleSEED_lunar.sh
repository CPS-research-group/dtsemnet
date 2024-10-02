#!/bin/bash


python -u train/train.py \
    --env_name lunar \
    --policy_type drnet \
    --train_name $2 \
    --seed $1 \
    --num_leaves $3 \
    --lr $4 \
    --ddt_lr $4 \
    --buffer_size 1000000 \
    --batch_size $5 \
    --gamma 0.99 \
    --tau 0.01 \
    --learning_starts 10000 \
    --eval_freq 1500 \
    --min_reward 250 \
    --training_steps 500000 \
    --log_interval 50 \
    --lin_control \
    --use_individual_alpha \
    --train_freq 64
  

# ./train_drnet_singleSEED_lunar.sh 11 16test 16 0.001 256