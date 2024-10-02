# Created by Yaru Niu

import gym
import numpy as np
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
import pandas as pd
from tqdm import tqdm
import time



from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy

# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import datetime

# ignore gym/stablebaseline warnings
import warnings
import os
warnings.filterwarnings('ignore', category=UserWarning)
torch.use_deterministic_algorithms(True)

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
        
    else:
        raise Exception('No valid environment selected')
    
    
    return env, name




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ICCT Testing')
    parser.add_argument('--env_name', help='environment to run on', type=str, default='lunar')
    parser.add_argument('--policy_type', help='policy type to test', type=str, default='drnet')
    parser.add_argument('--train_name', help='which model file to load and test', type=str, default='best_model')
    parser.add_argument('--render', help='if render the tested environment', action='store_true')
    parser.add_argument('--gpu', help='if run on a GPU', action='store_true')
    

    args = parser.parse_args()
    # if args.policy_type == 'dgt':
    #     args.policy_type = 'dgt_reg'
    if args.gpu:
        args.device = 'cuda'
    else:
        args.device = 'cpu'

    log_name = f'{args.env_name}_{"gpu" if args.gpu else "cpu"}/{args.policy_type}/{args.train_name}/'
    trained_model_dir = f"./trained_models/{log_name}"
    
    file_names = [file_name for file_name in os.listdir(trained_model_dir) if file_name.endswith('.zip')]
    
    print("=================> Testing <=================")
    rew_mean_array = np.array([])
    rew_std_array = np.array([])
    
    for file_name in file_names:
        # load model
        model = SAC.load(f'{trained_model_dir}{file_name}', device=args.device)
        
        test_reward_array = np.array([])
        # calculate mean reward over 100 episodes
        for seed in range(500, 600):
            # ========== Seed for Test Envs [500, 501 .... 599] ==========
            model.set_random_seed(seed)
            np.random.seed(seed)
            env, env_n = make_env(args.env_name, seed=seed)
            
            reward, _ = evaluate_policy(model,
                                        env,
                                        n_eval_episodes=1,
                                        deterministic=True)
            test_reward_array = np.append(test_reward_array, reward)

        test_reward_mean = np.mean(test_reward_array)
        test_reward_std = np.std(test_reward_array)
        rew_mean_array = np.append(rew_mean_array, test_reward_mean)
        rew_std_array = np.append(rew_std_array, test_reward_std)
        print(f'{file_name}: {test_reward_mean} +/- {test_reward_std}')
        print(f'{file_name}: {np.min(test_reward_array)}')
        

    overall_mean = np.mean(rew_mean_array)
    overall_std = np.std(rew_mean_array, ddof=1)

    # Calculate the square root of the sample size
    sample_size = len(rew_mean_array)
    sqrt_sample_size = np.sqrt(sample_size)
    # Calculate the standard error of the mean (SEM)
    sem = overall_std / sqrt_sample_size
    
    print(f'overall: mean: {overall_mean}, std:{overall_std}, sem:{sem}')
    # store rew_mean_array and rew_std_array in txt file at model_path
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    with open(f'{trained_model_dir}test_results_{timestamp}.txt', 'w') as f:
        f.write(f'env: {args.env_name}\n')
        f.write(f'model_path: {trained_model_dir} \n\n')
        f.write(f'file_names: {str(file_names)} \n\n')
        f.write(f'#### EVAL PAPER SUBMISSION ####\n')
        for i in range(len(rew_mean_array)):
            f.write(f'{np.round(rew_mean_array[i], 1)} +/- {np.round(rew_std_array[i], 1)} \n')
        f.write(f'-------[mean, sample std, standard error of mean]---------\n')
        f.write(f'{np.round(overall_mean, 2)}+/-{np.round(sem, 2)}, std:{np.round(overall_std, 2)}\n')