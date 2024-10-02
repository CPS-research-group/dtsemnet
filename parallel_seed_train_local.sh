#!/bin/bash

MODEL="drnet"
ENV="lunar"

# seeds=(11 13)
# seeds=(40 50 60 70 80)
# seeds=(11 13 17 19 23)
seeds=(11 13 17)


NUM_LEAF=64
CURRENT_DATE=$(date +'%M-%H-%d-%b')
TRAIN_NAME="${NUM_LEAF}leaves_${CURRENT_DATE}_def"
LR=0.001
BATCH_SIZE=256


# Loop over 5 different seeds in parallel
# ./train_drnet_singleSEED_lunar.sh 11 16exp 16 0.001 256
for seed in "${seeds[@]}"; do
    echo ">>Submitting with SEED $seed ..."
    bash "train_${MODEL}_singleSEED_${ENV}.sh" "${seed}" "${TRAIN_NAME}" "${NUM_LEAF}" "${LR}" "${BATCH_SIZE}" > /dev/null 2>&1 &   

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "done"
    else
        echo "Error: Failed for seed $seed with exit code $exit_code"
        exit 1
    fi

    sleep 2

done