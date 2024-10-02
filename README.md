# DTSemNets
RL experiments with continous actions.

## System Information
- OS: Ubuntu 22.04
- CPU: Intel Xeon W-2235 CPU @ 3.80GHz
- RAM: 32GB
- GPU: NVIDIA Quadro P620 2GB
- Python: 3.8.10
- Environment: conda


## General Installation (15-20mins)
Install package in same directory (installs gym, torch, stablebaselines3 etc.). Please install [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) if you don't have it already. 
```python 
conda env create -f environment.yml # creates conda environment
conda activate dtregnet # activates conda environment
```

## Train Agents
Update the `parallel_seed_train_local.sh' with correct option then run the script:
- MODEL='drnet' or 'dgt'
- ENV='lunar' or 'walker'

```bash
./parallel_seed_train_local.sh
```

If requried to execute the training for single seed:
```bash
.\train_drnet_singleSEED_lunar.sh 11 16test 16 0.001 256
```


## Eval Agent
Update the path of the model to be updated in `eval_agent.sh' then execute the script
```bash
.\eval_agent.sh
