'''Evaluate the trained agents from that were trained with stable-baselines3'''
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import random
import numpy as np
import os
from agents.dtnet.ppo_policy import DTSemNetACAgent


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""



if __name__ == "__main__":

    #==================#
    # Specify the stored model
    AGENT_TYPE = 'dtnet'  # 'dtnet'
    ENV_TYPE = 'cart'
    USE_GPU = False
    torch.use_deterministic_algorithms(True)
    MODEL_PATH = 'trained_models/cart_cpu/dtnet/aaai_v2_new_rand/' #'trained_models/cart_cpu/dtnet/aaai_15n_16x16c_rand/'
    #==================#


    # Define Environments and Hyperparameters
    if ENV_TYPE == 'cart':
        env = gym.make('CartPole-v1')
        seed_list = [62, 65, 100, 104, 112]
    elif ENV_TYPE == 'acrobot':
        env = gym.make('Acrobot-v1')
        seed_list = [42, 46, 48, 50, 51]

    elif ENV_TYPE == 'lunar':
        env = gym.make('LunarLander-v2')
        seed_list = [42, 46, 48, 50, 51]


    num_seeds = 5 # for each trained agent from 5 different seeds
    numbers = np.array([])
    print(f"===================> Testing <=================") # Print the seed
    for i in seed_list:
        # Evaluate the agent and store it the rewards
        # evaluate 30 episodes and report mean reward with std
        model = PPO.load(f"{MODEL_PATH}seed{i}",
                         policy=DTSemNetACAgent,
                         env=env,
                         device='cuda' if USE_GPU else 'cpu')

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
        print(f'---> Test reward for seeds [500, 600]: {test_reward_mean:.2f} +/- {test_reward_std} <---')
        numbers = np.append(numbers, test_reward_mean)
print('=================> Overall Mean and SME <==================')
mean = np.mean(numbers)
overall_std = np.std(numbers, ddof=1)
# Calculate the square root of the sample size
sample_size = len(numbers)
sqrt_sample_size = np.sqrt(sample_size)
# Calculate the standard error of the mean (SEM)
sem = overall_std / sqrt_sample_size
print(f"Mean: {mean:.2f} +/- {sem:.2f}")
