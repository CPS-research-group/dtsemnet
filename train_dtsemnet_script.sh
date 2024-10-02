#!/bin/bash

python ./train_agent/train_sb_gym_agent.py \
    -a fcnn \
    -env $2 \
    -e 1500 \
    -seed $1 \
    -log_name $3 \
    -an '32x32' \
    -cn '64x64' \
    -rand
