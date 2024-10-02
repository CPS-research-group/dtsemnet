# %%
# Created by Yaru Niu

import gym
import numpy as np
import time
import copy
import argparse
import random
import os
import torch
from icct.rl_helpers import ddt_sac_policy
from icct.core.icct_helpers import convert_to_crisp
from icct.rl_helpers.eval_callback import EpCheckPointCallback
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor
)

from stable_baselines3 import SAC
import highway_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy

# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import env

# ignore gym/stablebaseline warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def make_env(env_name, seed):
    set_random_seed(seed)
    if env_name == 'lunar':
        env = gym.make('LunarLanderContinuous-v2')
        name = 'LunarLanderContinuous-v2'
        env.seed(seed)
    elif env_name == 'walker':
        env = gym.make("BipedalWalker-v3")
        name = 'BipedalWalker-v3'
        env.seed(seed)
    elif env_name == 'cart':
        env = gym.make('InvertedPendulum-v2')
        name = 'InvertedPendulum-v2'
        env.seed(seed)
    elif env_name == 'lane_keeping':
        env = gym.make('lane-keeping-v0')
        name = 'lane-keeping-v0'
        env.seed(seed)
    elif env_name == 'ring_accel':
        create_env, gym_name = make_create_env(params=ring_accel_params, version=0)
        env = create_env()  
        name = gym_name
        env.seed(seed)

    else:
        raise Exception('No valid environment selected')
    
    
    return env, name
torch.use_deterministic_algorithms(True)

# %%

# load model
model_drnet = SAC.load(f'{"trained_models/walker_gpu/drnet/256leaves_18-22-10-Dec/"}best_model_seed{"11"}', device='cpu')
model_nn = SAC.load(f'{"trained_models/walker_cpu/mlp/256leaves_09-22-10-Dec/"}best_model_seed{"11"}.zip', device='cpu')

# %%
model_nn.policy.actor


num_params = sum(p.numel() for p in model_drnet.policy.actor.parameters())
print("Number of tunable parameters:", num_params)


# ========== Seed for Test Envs [500, 501 .... 599] ==========
model_drnet.set_random_seed(500)
model_nn.set_random_seed(500)
env, env_n = make_env('walker', seed=500)

# Set the batch size
batch_size = 1

# Create an empty list to store the samples
samples = []

# Record the inference time
start_time = time.time()

# Take random samples from the environment
for _ in range(batch_size):
    sample = env.observation_space.sample()
    samples.append(sample)

# Pass the samples through the policy network
for sample in samples:
    model_nn.predict(sample)

# Calculate the inference time
inference_time = time.time() - start_time

print("Inference time:", inference_time)

# %% Cound number of FLOPS
from fvcore.nn import FlopCountAnalysis
flops = FlopCountAnalysis(model_nn.policy.actor, sample)
print(flops.total())
print(flops.by_operator())



