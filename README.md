# DTSemNets: Decision Tree Semantic Networks
This repository implements the work proposed in "Gradient Descent on Decision Trees for Trustworthy Reinforcement Learning," which is exact conversion between DT and NN. The architecture proposed (main contribution) in this paper is at `src/agents/dtnet/arch`.

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
conda activate dtnet-rl    # activates conda environment
```

If modules in `/src` couldnot be imported (it should be installed automatically), manually install the src folder as a package with `python -m pip install -e .` inside the conda env `dtnet`. If GPU is not available, donot use `-gpu` flag in the scripts. 



## Project Structure
Training and testing scripts are in `scripts` which helps in training and evaluation of agents. Whereas, `src` contains all the architecture and learning algorithms.

```
dtnet
│   README.md
│   environment.yml -- for installing packages in conda environment    
│
└───scripts -- **contains script for training/testing agents**
│   │   train_sb_gym_agent.py 
│   
│───logs: Training logs
│───trained_models: Trained agents are stores here
│───results: Plots 
│───notebooks: Jupyter notebooks for various analysis on results
│   
└───src -- *libraries* which includes agent architectures and learning algorithms
    │───agents -- *various RL agents*
    │   │───dtnet -- *DTNet agent*
    │   │   │───agent.py
    │   │   │───arch.py
    │   │ 
    │   │───prolonet -- *Prolonet agent*
    │   │  
    │   │───dt -- *DT agent*
    │   │
    │───learner -- *various learning algorithms*
    │   │───ppo -- *PPO algorithm*
    │
    │───monitor -- *monitoring tools*

```


## Train Agent
Agents are trained with PPO implementation of stablebaselines3 and also with PPO implementation of ProLoNet.

```python
python ./train_agent/train_sb_gym_agent.py \
    -a fcnn \
    -env lunar \
    -e 1500 \
    -seed 42 \
    -log_name test \
    -an '32x32' \
    -cn '64x64' \
    -rand
```

-a: can be `dtnet`, `fcnn`, `prolo`

-env: can be `cart`, `lunar`, `acrobot`

-e: is number of episodes

-seed: is random seed

-log_name: is name of log file

-an: is architecture of actor network (applicable to FCNN)

-cn: is architecture of critic network (applicable to all agents)

-gpu: For running on GPU

-rand: For random initialization of DTNet agent

### For training multiple seeds parallelly
Please edit the log name and environment name in the script `parallel_seed_train_dtsemnet_script.sh` and run the script. Edit the train_dtsemnet_script.sh to change the agent.
```bash
./parallel_seed_train_dtsemnet_script.sh
```



