# DTSemNet-Topk

1. DTSemNet Top-k Regression
2. RL experiments with continous actions.

## General Installation (15-20mins)
Install package in same directory (installs gym, torch, stablebaselines3 etc.). Please install [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) if you don't have it already. 
```python 
conda env create -f environment.yml # creates conda environment
conda activate dtsemnet # activates conda environment
```

## Dataset
One datasets is given with the repo due to size constraint, others needs to downloaded and put into the /dataset directory. Please download it from [HuggingFace](https://huggingface.co/datasets/subratpp/dtsemnet-topk).

## Regression

Common flags:
    . --model dtregnet_topk or dtregnet_ste or dgt or cart
    . --dataset name under ./dataset/
    . -s Number of simulations (different seeds, used 10 seeds for the experiments)
    . --output_prefix prefix for logs
    . --verbose print training logs
    . -g use GPU if available

### Regression DTSemNet-Topk

```bash
python ./train/regression_train_topk.py --model dtregnet_topk --dataset ctslice -s 10 --output_prefix leaves_ct --verbose True -g
``` 

### Regression DTSemNet-ste

```bash
python ./train/regression_train_ste.py --model dtregnet_ste --dataset ctslice -s 10 --output_prefix leaves_ct --verbose True -g 
```

### Regression DGT

```bash
python ./train/regression_train_ste.py --model dgt --dataset ctslice -s 10 --output_prefix leaves_ct --verbose True -g > dgt_ct_leaf.log 
```

### Regression Cart

```bash
python ./train/regression_train_cart.py --model cart --dataset ctslice -s 10 --output_prefix leaves_ct --verbose True -g > cart_ct_leaf.log
```

## Train RL Agents

```bash
python train/clean_rl_sac_train.py --config configs/rl/dtsemnet_topk/lunar_cont.json --seed 11
```

OR

```bash
./parallel_seed_train_local.sh
```
