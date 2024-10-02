"""Train a policy on a given environment using Stable Baselines3 SAC.
Updated the code from CORERL-ICCT: https://github.com/CORE-Robotics-Lab/ICCT.git
"""
import logging
logger = logging.getLogger(__name__)
import datetime
import gym
import gymnasium
import numpy as np
import copy
import argparse
import random
import os
import torch
import time
import pandas as pd

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor
)

from constants import *

#info: Registers the custom policies
from icct.rl_helpers import ddt_sac_policy
from dtregnet.policies import dtregnet_sac_policy
from dgt import dgt_sac_policy, dgt_reg_sac_policy
from icct.rl_helpers.eval_callback import EpCheckPointCallback
from icct.rl_helpers.sac import SAC



# set cublas to deterministic mode (for reproducibility)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def make_env(env_name, seed):
    set_random_seed(seed)
    if env_name == 'lunar':
        env = gym.make('LunarLanderContinuous-v2')
        eval_env = gym.make('LunarLanderContinuous-v2')
        name = 'LunarLanderContinuous-v2'
        env.seed(seed)
        eval_env.seed(seed+5)
    elif env_name == 'walker':
        env = gym.make("BipedalWalker-v3") #hardcore=True
        eval_env = gym.make("BipedalWalker-v3")
        name = 'BipedalWalker-v3'
        env.seed(seed)
        eval_env.seed(seed+5)
    elif env_name == 'cart':
        env = gym.make('InvertedPendulum-v2')
        name = 'InvertedPendulum-v2'
        eval_env = gym.make('InvertedPendulum-v2')
        # # Print the observation space
        # print("Observation space:", env.observation_space)
        # print("Action space:", env.action_space)
        # exit()
        env.seed(seed)
        eval_env.seed(seed+5)
        
    else:
        raise Exception('No valid environment selected')
    
    
    return env, eval_env, name



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DTRegNet Training')
    parser.add_argument('--env_name', help='environment to run on', type=str, default='lunar')
    parser.add_argument('--policy_type', help='mlp or ddt', type=str, default='ddt')
    parser.add_argument('--train_name', help='the name of the log file', type=str, default='test')
    parser.add_argument('--mlp_size', help='the size of mlp (small|medium|large)', type=str, default='medium')
    parser.add_argument('--seed', help='the seed number to use', type=int, default=42)
    parser.add_argument('--num_leaves', help='number of leaves used in ddt (2^n)', type=int, default=16)
    parser.add_argument('--submodels', help='if use sub-models in ddt', action='store_true', default=False)
    parser.add_argument('--sparse_submodel_type', help='the type of the sparse submodel, 1 for L1 regularization, 2 for feature selection, other values for not sparse', type=int, default=0)
    parser.add_argument('--hard_node', help='if use differentiable crispification', action='store_true', default=False)
    parser.add_argument('--gpu', help='if run on a GPU', action='store_true', default=False)
    parser.add_argument('--lin_control', help='use linear controller at leaf', action='store_true', default=False)
    parser.add_argument('--hk', help='use human knowledge', action='store_true', default=False)
    parser.add_argument('--lr', help='learning rate', type=float, default=3e-4)
    parser.add_argument('--buffer_size', help='buffer size', type=int, default=1000000)
    parser.add_argument('--batch_size', help='batch size', type=int, default=256)
    parser.add_argument('--gamma', help='the discount factor', type=float, default=0.9999)
    parser.add_argument('--tau', help='the soft update coefficient (between 0 and 1)', type=float, default=0.01)
    parser.add_argument('--learning_starts', help='how many steps of the model to collect transitions for before learning starts', type=int, default=10000)
    parser.add_argument('--training_steps', help='total steps for training the model', type=int, default=500000)
    parser.add_argument('--argmax_tau', help='the temperature of the diff_argmax function', type=float, default=1.0)
    parser.add_argument('--ddt_lr', help='the learning rate of the ddt', type=float, default=3e-4)
    parser.add_argument('--use_individual_alpha', help='if use different alphas for different nodes', action='store_true', default=False)
    parser.add_argument('--l1_reg_coeff', help='the coefficient of the l1 regularization when using l1-reg submodels', type=float, default=5e-3)
    parser.add_argument('--l1_reg_bias', help='if consider biases in the l1 loss when using l1-reg submodels', action='store_true', default=False)
    parser.add_argument('--l1_hard_attn', help='if only sample one linear controller to perform L1 regularization for each update when using l1-reg submodels', action='store_true', default=False)
    parser.add_argument('--num_sub_features', help='the number of chosen features for submodels', type=int, default=1)
    parser.add_argument('--use_gumbel_softmax', help='if use gumble softmax instead of the differentiable argmax proposed in the paper', action='store_true', default=False)
    # evaluation and model saving
    parser.add_argument('--min_reward', help='minimum reward to save the model', type=int, default=200)
    parser.add_argument('--n_eval_episodes', help='the number of episodes for each evaluation during training', type=int, default=5)
    parser.add_argument('--eval_freq', help='evaluation frequence of the model', type=int, default=1500)
    parser.add_argument('--train_freq', help='train frequence of the model', type=int, default=1)
    parser.add_argument('--log_interval', help='the number of episodes before logging', type=int, default=4)

    #info: 0. Parse the arguments and Setup Logging
    args = parser.parse_args()
    training_name = args.train_name
    log_name = f'{args.env_name}_{"gpu" if args.gpu else "cpu"}/{args.policy_type}/{training_name}/'
    log_dir = f"./logs/{log_name}"
    trained_model_dir = f"./trained_models/{log_name}"
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    # Create model directory
    os.makedirs(trained_model_dir, exist_ok=True)

    # Configure Logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s[%(name)s]%(levelname)s | %(message)s',
                        datefmt='%d-%m-%y %H:%M',
                        handlers=[
                            logging.FileHandler(
                                f'{log_dir}training_seed{args.seed}.log', mode='w'),
                            logging.StreamHandler()
                        ])
    
    logging.info(f"Training on env: {args.env_name}")
    logging.info(f"Agent Type: {args.policy_type}" )
    logging.info(f"Log Directory: {log_dir}")
    logging.info(f"GPU?: {args.gpu}")
    logging.info(f"Num Leaves: {args.num_leaves}")

    
    #info: 0. Create the environment and setup SEED
    # SEED for reproducibility
    env, eval_env, env_n = make_env(args.env_name, args.seed)
    
    #info: 1. Set the seed for all the libraries
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    

    
    #info: 2. Instantiate the agent and learning algorithm
    args.alg_type = 'sac'
    args.tf = args.train_freq # sac training details
    args.gs = args.train_freq 


    #info: DDT (ICCT) and different versions
    if args.policy_type == 'ddt':
        if not args.submodels and not args.hard_node:
            logging.info('ICCT Policy Version: m1[cddt]')
            method = 'm1'
        elif args.submodels and not args.hard_node:
            logging.info('ICCT Policy Version: m2[cddt-complete]')
            method = 'm2'
            if args.sparse_submodel_type == 1 or args.sparse_submodel_type == 2:
                raise Exception('Not a method we want to test')
        elif not args.submodels and args.hard_node:
            logging.info('ICCT Policy Version: m3[icct-static: no controller]')
            method = 'm3'    
        else:
            if args.sparse_submodel_type != 1 and args.sparse_submodel_type != 2:
                logging.info('ICCT Policy Version: m4[icct-complete]')
                method = 'm4'
            elif args.sparse_submodel_type == 1:
                logging.info('ICCT Policy Version: m5a[icct-L1-sparse]')
                method = 'm5a'
            else:
                method = f'm5b_{args.num_sub_features}'
    
    #info: mlp (Linear network)
    elif args.policy_type == 'mlp':
        if args.mlp_size == 'small':
            method = 'mlp_s'
        elif args.mlp_size == 'medium':
            method = 'mlp_m'
        elif args.mlp_size == 'large':
            method = 'mlp_l'
        else:
            raise Exception('Not a valid MLP size')
    
    #info: DTRegNet Policy
    elif args.policy_type == 'drnet':
        logging.info('DTRegNet Policy')
    
    elif args.policy_type == 'dgt':
        logging.info('DGT Policy')
    
    elif args.policy_type == 'dgt_reg':
        logging.info('DGT Reg Policy')
    
    else:
        raise Exception('Not a valid policy type')
    
    # info: File Names for Monitor Logging
    monitor_file_path = log_dir + f'train_seed{args.seed}'
    env = Monitor(env, monitor_file_path)
    eval_monitor_file_path = log_dir + f'eval_seed{args.seed}'
    eval_env = Monitor(eval_env, eval_monitor_file_path)
    # INFO
    ev_callback = EpCheckPointCallback(eval_env=eval_env, best_model_save_path=trained_model_dir, n_eval_episodes=args.n_eval_episodes, log_path=log_dir,
                                    eval_freq=args.eval_freq, minimum_reward=args.min_reward, seed=args.seed)
    
    
    if args.gpu:
        args.device = 'cuda'
    else:
        args.device = 'cpu'
        
    if args.env_name == 'lane_keeping':
        features_extractor = CombinedExtractor
    else:
        features_extractor = FlattenExtractor

    if args.env_name == 'cart' and args.policy_type == 'ddt':
        args.fs_submodel_version = 1 # some chages in architecture
    else:
        args.fs_submodel_version = 0
    
    if args.policy_type == 'ddt':
        ddt_kwargs = {
            'num_leaves': args.num_leaves,
            'submodels': args.submodels,
            'hard_node': args.hard_node,
            'device': args.device,
            'argmax_tau': args.argmax_tau,
            'ddt_lr': args.ddt_lr,
            'use_individual_alpha': args.use_individual_alpha,
            'sparse_submodel_type': args.sparse_submodel_type,
            'fs_submodel_version': args.fs_submodel_version,
            'l1_reg_coeff': args.l1_reg_coeff,
            'l1_reg_bias': args.l1_reg_bias,
            'l1_hard_attn': args.l1_hard_attn,
            'num_sub_features': args.num_sub_features,
            'use_gumbel_softmax': args.use_gumbel_softmax,
            'alg_type': args.alg_type
        }
        policy_kwargs = {
            'features_extractor_class': features_extractor,
            'ddt_kwargs': ddt_kwargs
        }
        if args.alg_type == 'sac':
            policy_name = 'DDT_SACPolicy'
            if args.env_name == 'walker':
                policy_kwargs['net_arch'] = {'pi': [2, 2], 'qf': [512, 512]}
            else:
                policy_kwargs['net_arch'] = {'pi': [2, 2], 'qf': [256, 256]} # [256, 256] is a default setting in SB3 for SAC
        
    elif args.policy_type == 'drnet':
        drnet_kwargs = {
            'num_leaves': args.num_leaves,
            'env_name': args.env_name,
            'device': args.device,
            'argmax_tau': args.argmax_tau,
            'ddt_lr': args.ddt_lr,
            'use_individual_alpha': args.use_individual_alpha,
            'fs_submodel_version': args.fs_submodel_version,
            'num_sub_features': args.num_sub_features,
            'use_gumbel_softmax': args.use_gumbel_softmax,
            'alg_type': args.alg_type,
            'lin_control': args.lin_control,
            'hk': args.hk
        }
        policy_kwargs = {
            'features_extractor_class': features_extractor,
            'drnet_kwargs': drnet_kwargs
        }
        if args.alg_type == 'sac':
            policy_name = 'DTRegNet_SACPolicy'
            #why: Why policy arg is required here?
            # for value network
            if args.env_name == 'walker':
                policy_kwargs['net_arch'] = {'pi': [2, 2], 'qf': [512, 512]}
            else:
                policy_kwargs['net_arch'] = {'pi': [2, 2], 'qf': [256, 256]} # [256, 256] is a default setting in SB3 for SAC
    
    elif args.policy_type == 'dgt':
        drnet_kwargs = {
            'num_leaves': args.num_leaves,
            'env_name': args.env_name,
            'device': args.device,
            'argmax_tau': args.argmax_tau,
            'ddt_lr': args.ddt_lr,
            'use_individual_alpha': args.use_individual_alpha,
            'fs_submodel_version': args.fs_submodel_version,
            'num_sub_features': args.num_sub_features,
            'use_gumbel_softmax': args.use_gumbel_softmax,
            'alg_type': args.alg_type
        }
        policy_kwargs = {
            'features_extractor_class': features_extractor,
            'drnet_kwargs': drnet_kwargs
        }
        if args.alg_type == 'sac':
            policy_name = 'DGT_SACPolicy'
            #why: Why policy arg is required here?
            # for value network
            policy_kwargs['net_arch'] = {'pi': [2, 2], 'qf': [256, 256]} # [256, 256] is a default setting in SB3 for SAC
    
    elif args.policy_type == 'dgt_reg':
        drnet_kwargs = {
            'num_leaves': args.num_leaves,
            'env_name': args.env_name,
            'device': args.device,
            'argmax_tau': args.argmax_tau,
            'ddt_lr': args.ddt_lr,
            'use_individual_alpha': args.use_individual_alpha,
            'fs_submodel_version': args.fs_submodel_version,
            'num_sub_features': args.num_sub_features,
            'use_gumbel_softmax': args.use_gumbel_softmax,
            'alg_type': args.alg_type
        }
        policy_kwargs = {
            'features_extractor_class': features_extractor,
            'drnet_kwargs': drnet_kwargs
        }
        if args.alg_type == 'sac':
            policy_name = 'DGTreg_SACPolicy'
            #why: Why policy arg is required here?
            # for value network
            if args.env_name == 'walker':
                policy_kwargs['net_arch'] = {'pi': [2, 2], 'qf': [256, 256]}
            else:
                policy_kwargs['net_arch'] = {'pi': [2, 2], 'qf': [256, 256]} # [256, 256] is a default setting in SB3 for SAC
                 
    elif args.policy_type == 'mlp':
        if args.env_name == 'lane_keeping':
            policy_name = 'MultiInputPolicy'
        else:
            policy_name = 'MlpPolicy'
            
        if args.mlp_size == 'small':
            if args.env_name == 'cart':
                pi_size = [6, 6]
            elif args.env_name in ['lunar', 'highway', 'intersection', 'racetrack']:
                pi_size = [8, 8]
            elif args.env_name == 'lane_keeping':
                pi_size = [6, 6]
            elif args.env_name == 'ring_accel':
                pi_size = [3, 3]
            elif args.env_name == 'ring_lane_changing':
                pi_size = [3, 3] 
            elif args.env_name == 'walker':
                pi_size = [74, 74] 
            else:
                pi_size = [16, 16] 
        elif args.mlp_size == 'medium':
            if args.env_name == 'cart':
                pi_size = [8, 8]
            elif args.env_name in ['lunar', 'highway', 'intersection', 'racetrack']: 
                pi_size = [16, 16]
            elif args.env_name == 'lane_keeping':
                pi_size = [14, 14]
            elif args.env_name == 'ring_accel':
                pi_size = [12, 12]
            elif args.env_name == 'ring_lane_changing':
                pi_size = [32, 32] 
            elif args.env_name == 'walker':
                pi_size = [110, 110] 
            else:
                pi_size = [32, 32] 
        elif args.mlp_size == 'large':
            if args.env_name in ['lunar', 'highway', 'intersection', 'racetrack']: 
                pi_size = [32, 32]
            elif args.env_name == 'walker':
                pi_size = [160, 160]
            elif args.env_name == 'ugrid':
                pi_size = [128, 128]
            else:
                pi_size = [64, 64]
            
        else:
            raise Exception('Not a valid MLP size')
        logging.info(f"MLP Size: {pi_size}")
        if args.alg_type == 'sac':
            if args.env_name == 'walker':
                policy_kwargs = {
                    'net_arch': {'pi': pi_size, 'qf': [512, 512]},
                    'features_extractor_class': features_extractor,
                }
                
            else:
                policy_kwargs = {
                    'net_arch': {'pi': pi_size, 'qf': [256, 256]},
                    'features_extractor_class': features_extractor,
                }
        else:
            policy_kwargs = {
                'net_arch': {'pi': pi_size, 'qf': [400, 300]},
                'features_extractor_class': features_extractor,
            }
    else:
        raise Exception('Not a valid policy type')
    logging.info(f"Value Net Size: {policy_kwargs['net_arch']['qf']}")
    args.vnet = str(policy_kwargs['net_arch']['qf'])
    
    # Store Hyperparameters
    hyperparams = vars(args)
    #store the hyperparameters in the log directory as text file
    with open(f'{log_dir}hyperparams.txt', 'w') as f:
        for key, value in hyperparams.items():
            f.write(f'{key}: {value}\n')
    
    # log start training
    # Generate a timestamp string in the format of "YYYY-MM-DD HH:MM:SS"
    timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"[START]======> Training Started for Seed [{args.seed}]: {timestamp_str}")
    # Start the timer
    start_time = time.time()

    if args.env_name == 'walker':
        model = SAC(policy_name, env,
                    learning_rate=args.lr,
                    buffer_size=args.buffer_size,
                    batch_size=args.batch_size,
                    ent_coef='auto',
                    train_freq=args.tf, # changed
                    gradient_steps=args.gs, # changed
                    gamma=args.gamma,
                    use_sde=False, # added #info: SDE is not correctly implemented for DTs
                    tau=args.tau,
                    learning_starts=args.learning_starts,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=f"{log_dir}/tensorboard/",
                    verbose=1,
                    device=args.device,
                    seed=args.seed)
    else:
        model = SAC(policy_name, env,
                learning_rate=args.lr,
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                ent_coef='auto',
                train_freq=1,
                gradient_steps=1,
                gamma=args.gamma,
                tau=args.tau,
                learning_starts=args.learning_starts,
                policy_kwargs=policy_kwargs,
                tensorboard_log=f"{log_dir}tensorboard/",
                verbose=1,
                device=args.device,
                seed=args.seed)
        
    
    model.learn(total_timesteps=args.training_steps, log_interval=args.log_interval, callback=ev_callback)

    # ==== Saving execution time
    # End the timer
    end_time = time.time()

    rew = np.array(ev_callback.episode_rewards) 
    test_rew = np.array(ev_callback.test_rewards)
    worst_rew = np.array(ev_callback.worst_rewards)

    
    fn = f"{log_dir}seed{args.seed}_rewards.txt"
    with open(fn, "w") as file:
        np.savetxt(fn, rew, fmt='%.1f')

    fn = f"{log_dir}seed{args.seed}_test_rewards.txt"
    with open(fn, "w") as file:
        np.savetxt(fn, test_rew, fmt='%.2f')

    fn = f"{log_dir}seed{args.seed}_worst_rewards.txt"
    with open(fn, "w") as file:
        np.savetxt(fn, worst_rew, fmt='%.2f')

    # Calculate the execution time in seconds
    execution_time = end_time - start_time
    # Convert execution time to hours, minutes, and seconds
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int(execution_time % 60)
    # Print the execution time
    logging.info(f"[END]======> Execution time of seed {args.seed}: {hours} hours, {minutes} minutes, {seconds} seconds")

