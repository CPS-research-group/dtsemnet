# %%
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import torch.nn as nn
import torch
import argparse
import random
import numpy as np
import os
from stable_baselines3.common.evaluation import evaluate_policy
from agents.dtnet.ppo_policy import DTSemNetACAgent


import logging

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

ENV_TYPE = 'lunar'
AGENT_NAME = 'icct'
RANDOM = True

# %%
import optuna
from stable_baselines3 import PPO


def objective(trial: optuna.Trial) -> float:
    # Define the hyperparameters to be tuned
    if ENV_TYPE == 'cart':
        env = gym.make('CartPole-v1')

    elif ENV_TYPE == 'lunar':
        env = gym.make('LunarLander-v2')

    elif ENV_TYPE == 'acrobot':
        env = gym.make('Acrobot-v1')
    elif ENV_TYPE == 'pend':
        env = gym.make('Pendulum-v1', g=9.81)
    elif ENV_TYPE == 'lunar_cont':
        env = gym.make('LunarLanderContinuous-v2')
    elif ENV_TYPE == 'walker':
        env = gym.make('BipedalWalker-v3')

    n_steps = trial.suggest_int("n_steps", 500, 4000)
    batch_size = trial.suggest_int("batch_size", 32, 256)
    gamma = trial.suggest_float("gamma", 0.99, 0.99)
    gae_lambda = trial.suggest_float("gae_lambda", 0.85, 0.98)
    clip_range = trial.suggest_float("clip_range", 0.15, 0.25)
    vf_coef = trial.suggest_float("vf_coef", 0.4, 0.6)
    ent_coef = trial.suggest_float("ent_coef", 0.00, 0.03)
    learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.1)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.2, 0.6)
    n_epochs = trial.suggest_int("n_epochs", 4, 40)

    # Create the PPO agent with the hyperparameters
    agent = PPO(
        DTSemNetACAgent,
        env,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        n_epochs=n_epochs,
        policy_kwargs=dict(env_name=ENV_TYPE,
                           agent_name=AGENT_NAME,
                           rand=RANDOM),
    )


    # Train the agent
    agent.learn(total_timesteps=50000)

    # Evaluate the agent and return the evaluation metric
    mean_reward, _ = evaluate_policy(agent,
                                     env,
                                     deterministic=True,
                                     n_eval_episodes=30)
    return mean_reward



# %%
if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
