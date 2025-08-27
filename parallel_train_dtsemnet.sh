#!/bin/bash

ENV_TYPE="lunar"
MODEL_TYPE="dtsemnet_topk"
CONFIG="${ENV_TYPE}_cont_top4"
seeds=(11 13 17 19 23)

# Create base output directory
TIMESTAMP=$(date +%y%m%d_%H%M%S)
OUTPUT_DIR="tmp/${ENV_TYPE}/${MODEL_TYPE}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Loop over seeds and run in background
for seed in "${seeds[@]}"; do
    echo ">> Submitting with SEED $seed ..."
    nohup python "train/clean_rl_sac_train.py" \
        --config "./configs/rl/${MODEL_TYPE}/${CONFIG}.json" \
        --seed "${seed}" \
        > "${OUTPUT_DIR}/${CONFIG}_${seed}.out" 2>&1 &

    sleep 2  # slight delay between jobs
done

echo "All jobs submitted in the background."
