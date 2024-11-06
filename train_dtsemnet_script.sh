#!/bin/bash

python ./train_agent/train_sb_gym_agent.py \
    -a dtnet \
    -env $2 \
    -e 1500 \
    -seed $1 \
    -log_name $3 \
    -an '16' \
    -cn '16x16' \
    -rand
