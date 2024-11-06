#!/bin/bash

LOG_NAME="new"
ENV_TYPE="cart"

case "$ENV_TYPE" in
    "cart")
        seeds=(100 62 65 104 112)
        ;;
    "lunar")
        seeds=(11 13 17 19 23)
        ;;
    "acrobot")
        seeds=(100 105 110 115 120)
        ;;
    *)
        echo "Unknown environment type"
        exit 1
        ;;
esac

# Loop over 5 different seeds in parallel
for seed in "${seeds[@]}"; do
    echo ">>Submitting with SEED $seed ..."
    mkdir -p "tmp/${ENV_TYPE}/"
    nohup bash "train_dtsemnet_script.sh" "${seed}" "${ENV_TYPE}" "${LOG_NAME}" > "tmp/${ENV_TYPE}/${LOG_NAME}_nohup_${seed}.out" 2>&1 &   

    exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "done"
    else
        echo "Error: Failed for seed $seed with exit code $exit_code"
        exit 1
    fi

    sleep 2

done

echo "All processes checked and sent to the background."