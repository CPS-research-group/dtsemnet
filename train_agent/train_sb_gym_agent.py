# %%
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import torch.nn as nn
import torch
import argparse
import random
import numpy as np
import os
import datetime
import json
from stable_baselines3.common.monitor import Monitor
from agents.icct.icct_helpers import convert_to_crisp

from agents.dtnet.ppo_policy import DTSemNetACAgent
from agents.sbagent import ACAgent

import logging

import time


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# %%
class EpisodeRewardCallback(BaseCallback):
    def __init__(self, eps, model, model_path, seed, verbose=0, env=None):
        super(EpisodeRewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.num_eps = 0
        self.eps = eps
        self.model_path = model_path
        self.seed = seed
        self.max_r_till_now = -np.inf
        self.agent = model
        self.env = env # for evaluation and storing of model

    def _on_step(self) -> bool:
        # Get the reward of the current step
        reward = self.locals["rewards"][0]

        # Update the current episode reward
        self.current_episode_reward += reward




        # Check if the episode is done
        if self.locals.get("dones"):
            # Append the episode reward to the list
            self.episode_rewards.append(self.current_episode_reward)
            # store best model by checking evaluation reward after 10 episodes
            if self.num_eps % 10 == 0:
                self.env.seed(self.seed+100) # use different seed for eval env
                eval_reward, _ = evaluate_policy(self.agent,
                                                    self.env,
                                                    deterministic=True,
                                                    n_eval_episodes=30)
                                            
                if eval_reward >= self.max_r_till_now and self.num_eps > 200: # only store model after 100 episodes
                    self.agent.save(f"{self.model_path}seed{self.seed}")
                    self.max_r_till_now = eval_reward
                    logging.debug(
                        f"===>Storing Best Model at Episode: {self.num_eps}, Max_now: {self.max_r_till_now}<===")

            # if np.mean(self.episode_rewards[-20:]) > self.max_r_till_now:
            #     self.agent.save(f"{self.model_path}seed{self.seed}")
            #     self.max_r_till_now = np.mean(self.episode_rewards[-20:])
            #     logging.debug(
            #         f"===>Storing Best Model at Episode: {self.num_eps}, Max_now: {self.max_r_till_now}<===")

            self.current_episode_reward = 0
            if self.num_eps % 50 == 0:
                # every 50 eps print avg reward of last 50 eps
                avg_reward = np.mean(self.episode_rewards[-100:])
                logging.info(f"Episode: {self.num_eps}, Avg Reward: {avg_reward}, Best Eval: {self.max_r_till_now}")

            self.num_eps += 1
            if self.num_eps == self.eps:
                #stop training
                return False

        return True


# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-a",
                        "--agent_type",
                        help="Agent Architecture",
                        type=str,
                        default='dtnet')
    parser.add_argument("-e",
                        "--episodes",
                        help="Number of episodes to run",
                        type=int,
                        default=1000)
    parser.add_argument("-env",
                        "--env_type",
                        help="environment to run on",
                        type=str,
                        default='cart')
    parser.add_argument("-gpu", help="run on GPU?", action='store_true')
    parser.add_argument("-rand", help="Random Weight Initialization", action='store_true')
    parser.add_argument("-seed", help="Random Seed", type=int, default=0)
    parser.add_argument("-log_name", help="Log Name", type=str, default='test')
    parser.add_argument("-an", help="Actor Network", type=str, default='32x32')
    parser.add_argument("-cn", help="Critic Network", type=str, default='32x32')

    args = parser.parse_args()

    AGENT_TYPE = args.agent_type  # 'dtnet'
    NUM_EPS = args.episodes  # num episodes Default 1000
    ENV_TYPE = args.env_type  # 'cart' or 'lunar' Default 'cart'
    USE_GPU = args.gpu  # Default false
    if not USE_GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    RANDOM = args.rand
    torch.use_deterministic_algorithms(True)


    # check user inputs
    assert AGENT_TYPE in ['dtnet', 'fcnn', 'dgt', 'icct'], "Invalid Agent Type"
    assert ENV_TYPE in ['cart', 'lunar', 'acrobot'], "Invalid Environment Type"

    # name appended to log directory
    lg_name = args.log_name

    log_dir = f'{ENV_TYPE}_{"gpu" if USE_GPU else "cpu"}/{AGENT_TYPE}/aaai_v2_{lg_name}{"_rand" if RANDOM else ""}/'
    # Create log directory
    if not os.path.exists(f'./logs/{log_dir}'):
        try:
            os.makedirs(f'./logs/{log_dir}')
        except:
            pass
    # Create model directory
    if not os.path.exists(f'./trained_models/{log_dir}'):
        try:
            os.makedirs(f'./trained_models/{log_dir}')
        except:
            print("Model Directory already exists")
            

    # Configure Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s[%(name)s]%(levelname)s | %(message)s',
        datefmt='%d-%m-%y %H:%M',
        handlers=[
            logging.FileHandler(f'./logs/{log_dir}training{args.seed}.log', mode='w'),
            logging.StreamHandler()
        ])
    logging.info(f"Training on {ENV_TYPE} environment for {NUM_EPS} episodes")
    logging.info(f"Agent Type: {AGENT_TYPE}" )
    logging.info(f"Log Directory: {log_dir}")
    logging.info(f"Random Weight Initialization: {RANDOM}")

    # Generate a timestamp string in the format of "YYYY-MM-DD HH:MM:SS"
    timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    logging.info(f"Training Started: {timestamp_str}")

    # Define Environments and Hyperparameters
    if ENV_TYPE == 'cart':
        logging.info('Using Cart Pole Environment')
        env = gym.make('CartPole-v1')
        env_eval = gym.make('CartPole-v1')
        TSTEPS = 600000
        if AGENT_TYPE == 'dtnet':

            hyperparams = {"learning_rate": 0.02,
                           "gamma": 0.99,
                           "batch_size": 32,
                           "n_steps": 1024,
                           "ent_coef": 0.008,
                           "clip_range": 0.2,
                           "vf_coef": 0.5,
                           "max_grad_norm": 0.5,
                           "gae_lambda": 0.96,
                           "n_epochs": 14}
            if RANDOM:
                hyperparams = {
                    "learning_rate": 0.024,
                    "gamma": 0.99,
                    "batch_size": 32,
                    "n_steps": 832,
                    "ent_coef": 0.02,
                    "clip_range": 0.2,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5,
                    "gae_lambda": 0.86,
                    "n_epochs": 20
                }

        elif AGENT_TYPE == 'fcnn':
            hyperparams = {
                "learning_rate": 0.015,
                "gamma": 0.99,
                "batch_size": 32,
                "n_steps": 512,
                "ent_coef": 0.008,
                "clip_range": 0.2,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "gae_lambda": 0.92,
                "n_epochs": 12
            }

        elif AGENT_TYPE == 'dgt':
            hyperparams = {
                'n_steps': 542,
                'batch_size': 43,
                'gamma': 0.99,
                'gae_lambda': 0.934,
                'clip_range': 0.174,
                'vf_coef': 0.40,
                'ent_coef': 0.02,
                'learning_rate': 0.0042,
                'max_grad_norm': 0.530,
                'n_epochs': 7
            }

        elif AGENT_TYPE == 'icct':
             hyperparams = {
                    "learning_rate": 0.024,
                    "gamma": 0.99,
                    "batch_size": 32,
                    "n_steps": 832,
                    "ent_coef": 0.02,
                    "clip_range": 0.2,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5,
                    "gae_lambda": 0.86,
                    "n_epochs": 20
                }

    elif ENV_TYPE == 'acrobot':
        logging.info('Using Acrobot Environment')
        env = gym.make('Acrobot-v1')
        env_eval = gym.make('Acrobot-v1')
        TSTEPS = 800000
        if AGENT_TYPE == 'dtnet':
            hyperparams = {
                'learning_rate': 0.018,
                'ent_coef': 0.02,
                'gamma': 0.99,
                'gae_lambda': 0.85,
                'clip_range': 0.2,
                'n_epochs': 13,
                'batch_size': 64,
                'n_steps': 1024,
                'vf_coef': 0.5,
                'max_grad_norm': 0.52,
            }
            if RANDOM:
                hyperparams['learning_rate'] = 0.045
                hyperparams['gae_lambda'] = 0.80
                hyperparams['ent_coef'] = 0.01
                hyperparams['n_epochs'] = 16

        elif AGENT_TYPE == 'fcnn':
            hyperparams = {
                'learning_rate': 0.007,
                'ent_coef': 0.02,
                'gamma': 0.99,
                'gae_lambda': 0.85,
                'clip_range': 0.2,
                'n_epochs': 16,
                'batch_size': 64,
                'n_steps': 1024,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
            }

        elif AGENT_TYPE == 'dgt':
            hyperparams = {
                'n_steps': 2048,
                'batch_size': 128,
                'gamma': 0.99,
                'gae_lambda': 0.94,
                'clip_range': 0.22,
                'vf_coef': 0.54,
                'ent_coef': 0.012,
                'learning_rate': 0.001,
                'max_grad_norm': 0.5,
                'n_epochs': 20
            }

        elif AGENT_TYPE == 'icct':
            hyperparams = {
                'learning_rate': 0.045,
                'ent_coef': 0.01,
                'gamma': 0.99,
                'gae_lambda': 0.80,
                'clip_range': 0.2,
                'n_epochs': 16,
                'batch_size': 64,
                'n_steps': 1024,
                'vf_coef': 0.5,
                'max_grad_norm': 0.52,
            }

    elif ENV_TYPE == 'lunar':
        logging.info('Using Lunar Lander Environment')
        env = gym.make('LunarLander-v2')
        env_eval = gym.make('LunarLander-v2')
        TSTEPS = 800000  # training steps
        if AGENT_TYPE == 'dtnet':

            hyperparams = {
                "learning_rate": 0.04,
                "ent_coef": 0.01,
                "gamma": 0.99,
                "gae_lambda": 0.9,
                "clip_range": 0.2,
                "n_epochs": 20,
                "batch_size": 64,
                "n_steps": 2048,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5
            }

            if RANDOM:
                hyperparams = {
                    "learning_rate": 0.02,
                    "ent_coef": 0.01,
                    "gamma": 0.99,
                    "gae_lambda": 0.9,
                    "clip_range": 0.2,
                    "n_epochs": 10,
                    "batch_size": 64,
                    "n_steps": 2048,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5
                }
                
        elif AGENT_TYPE == 'fcnn':
            hyperparams = {"learning_rate": 0.005,
                           "ent_coef": 0.01,
                           "gamma": 0.99,
                           "gae_lambda": 0.9,
                           "clip_range": 0.2,
                           "n_epochs": 10,
                           "batch_size": 64,
                           "n_steps": 2048}

        elif AGENT_TYPE == 'dgt':
            hyperparams = {
                    "learning_rate": 0.02,
                    "ent_coef": 0.01,
                    "gamma": 0.99,
                    "gae_lambda": 0.9,
                    "clip_range": 0.2,
                    "n_epochs": 4,
                    "batch_size": 64,
                    "n_steps": 2048,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5
                }

        elif AGENT_TYPE == 'icct':
            hyperparams = {
                    "learning_rate": 0.005,
                    "ent_coef": 0.01,
                    "gamma": 0.99,
                    "gae_lambda": 0.9,
                    "clip_range": 0.2,
                    "n_epochs": 4,
                    "batch_size": 64,
                    "n_steps": 2048,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5
                }



    # Log the JSON string
    json_str = json.dumps(hyperparams)
    logging.info('Hyperparameters: %s', json_str)






    # if REPRODUCE:
    # SPP: Make training deterministic
    seed = args.seed
    env.seed(seed)
    env_eval.seed(seed+100) # use different seed for eval env
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

    monitor_file_path =  f'./monitor/{log_dir}train_seed{seed}'
    env = Monitor(env, monitor_file_path)


    logging.info(f"===================> Training with seed {seed} <=================") # Print the seed
    # Start the timer
    start_time = time.time()

    policy_kwargs = dict(agent_name = AGENT_TYPE, 
                         env_name=ENV_TYPE, 
                         rand = RANDOM,
                         act_arch = args.an,
                         val_arch = args.cn) # network architecture (for fcnn and critic)
    # Create the PPO agent

    model = PPO(DTSemNetACAgent,
                env,
                verbose=1,
                seed=seed,
                device='cuda' if USE_GPU else 'cpu',
                policy_kwargs=policy_kwargs,
                tensorboard_log=f"logs/{log_dir}tensorboard/",
                **hyperparams)
    callback = EpisodeRewardCallback(eps=NUM_EPS, model=model, model_path = f'./trained_models/{log_dir}', seed=args.seed, env = env_eval)

    model.set_random_seed(seed)
    # Train the agent
    model.learn(total_timesteps=TSTEPS,
                callback=callback,
                log_interval=500)

    # Save the agent
    # model.save(f"trained_models/{log_dir}seed{i}")
    # Save the rewards
    rew = np.array(callback.episode_rewards)[:NUM_EPS] #only first 1000 episodes

    reward_fp = f'logs/{log_dir}'
    fn = f"{reward_fp}seed{args.seed}_rewards.txt"
    with open(fn, "w") as file:
        np.savetxt(fn, rew, fmt='%.1f')

    # %%
    # Evaluate the agent and store it the rewards
    # evaluate 100 episodes and report mean reward with std
    model = PPO.load(f"trained_models/{log_dir}seed{args.seed}",
                        policy=DTSemNetACAgent,
                        env=env,
                        device='cuda' if USE_GPU else 'cpu')
    
    # ======> discretize the agent of ICCT
    # if AGENT_TYPE == 'icct':
    #     model.policy.action_net = convert_to_crisp(model.policy.action_net)
    
    if not os.path.exists(f'results/{log_dir}'):
        os.makedirs(f'results/{log_dir}')

    test_reward_array = np.array([])
    # calculate mean reward over 100 episodes
    for i in range(500, 600):
        # ========== Seed for Test Envs [500, 501 .... 599] ==========
        model.set_random_seed(i)
        # env = Monitor(env, f'./results/{log_dir}logs/')
        env.seed(i)
        random.seed(i)
        np.random.seed(i)
        reward, _ = evaluate_policy(model,
                                    env,
                                    n_eval_episodes=1,
                                    deterministic=True)
        test_reward_array = np.append(test_reward_array, reward)

    test_reward_mean = np.mean(test_reward_array)
    test_reward_std = np.std(test_reward_array)

    # Store test results in txt file (deterministic=True)
    logging.info(f'==============> Test reward for seeds [500, 600]: {test_reward_mean:.2f} +/- {test_reward_std} <==========')

    # check and create eval directory
    if not os.path.exists(f'logs/{log_dir}eval'):
        os.makedirs(f'logs/{log_dir}eval')

    # Print the execution time
    test_rew_fp = f'logs/{log_dir}eval/test_rewards_seed{seed}.txt'
    with open(test_rew_fp, 'w') as test_rew_file:
        test_rew_file.write(f"\n{test_reward_mean:.3f}")

    # %% ========= Calculate execution time =========
    # End the timer
    end_time = time.time()

    # Calculate the execution time in seconds
    execution_time = end_time - start_time
    # Convert execution time to hours, minutes, and seconds
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int(execution_time % 60)


    logging.info(f"Execution time of seed {args.seed}: {hours} hours, {minutes} minutes, {seconds} seconds")
