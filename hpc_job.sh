#!/bin/bash
#SBATCH --job-name=dtnet
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=8           
#SBATCH --partition=NH100q          
#SBATCH --output=train_output_%j.out 

# Activate your virtual environment (if any)
eval "$(conda shell.bash hook)"
conda activate dtnet-ecai # source activate myenv

# Command to run your Python script
# python -m cro_dt.net_train --model dgt --dataset all --depth 3 -s 10 --output_prefix f --verbose True

python -m src.net_train --model dtsemnet --dataset all --depth 3 -s 100 --output_prefix f --verbose True

# python -m src.cart_train --model cart --dataset all --depth 3 -s 100 --output_prefix t --verbose True

# python -m src.reg_train --model dtregnet --dataset abalone -s 10 --output_prefix t --verbose True

# python -m src.cro_dt --dataset all --cro_config configs/simple_sa1.json --simulations 10 --depth 3 --initial_pop None --alpha 1.0 --should_normalize_rows False --should_cart_init True --should_normalize_dataset False --should_normalize_penalty False --should_get_best_from_validation False --should_apply_exponential False --should_use_threshold False --start_from 0 --evaluation_scheme matrix --output_prefix cro-dt-cs_depth3 --verbose True


python -m src.cro_dt --dataset mnist --cro_config configs/simple_sa1.json --simulations 1 --depth 8 --initial_pop None --alpha 1.0 --should_normalize_rows False --should_cart_init True --should_normalize_dataset False --should_normalize_penalty False --should_get_best_from_validation False --should_apply_exponential False --should_use_threshold False --start_from 0 --evaluation_scheme matrix --output_prefix cro-dt-cs_depth3 --verbose True