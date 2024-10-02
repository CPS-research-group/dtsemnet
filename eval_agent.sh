#!/bin/bash

ENV="walker"
MODEL="drnet"
TRAIN_NAME="64leaves_53-22-30-Sep_hk_abs2"

python -u train/test.py \
  --env_name ${ENV} \
  --model_path ./trained_models/${ENV}_cpu/${MODEL}/${TRAIN_NAME}/ \
  | tee test.log
