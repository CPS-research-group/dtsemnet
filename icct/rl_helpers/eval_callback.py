# Created by Yaru Niu and Andrew Silva

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import os
from typing import Any, Callable, Dict, List, Optional, Union
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import numpy as np
import warnings
import logging
from stable_baselines3.common.results_plotter import load_results, ts2xy
import csv


logger = logging.getLogger(__name__)

class EpCheckPointCallback(EvalCallback):
    """
    Callback for evaluating an agent. This callback is called with a certain frequency.

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every eval_freq call of the callback.
    :param log_path: Path to a folder where the evaluations
        will be saved. It will be updated at each evaluation.
    :param minimum_reward: The minimum reward to reach to save a model
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: str = None,
        minimum_reward: int = 200,
        best_model_save_path: str = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        seed: int = 0,
        warn: bool = True,
    ):
        super(EpCheckPointCallback, self).__init__(eval_env=eval_env,
                                                   callback_on_new_best=callback_on_new_best,
                                                   n_eval_episodes=n_eval_episodes,
                                                   eval_freq=eval_freq,
                                                   log_path=log_path,
                                                   best_model_save_path=best_model_save_path,
                                                   deterministic=deterministic,
                                                   render=render,
                                                   warn=warn,
                                                   verbose=verbose)
        # Minimun reward to save the model
        self.minimum_reward = minimum_reward
        self.seed = seed
        self.log_path = log_path
        self.episode_rewards = []
        self.test_rewards = []
        self.worst_rewards = []
        self.cumm_reward = 0
        self.num_eps = 0
        self.max_r_till_now = -np.inf
        
        
        # store exp weights, associated with y-velocity
        self.file_path_exp = os.path.join(self.log_path, f"exp_weight_{self.seed}.csv")
        with open(self.file_path_exp, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Weights"])
        
        # store normal weights for x-velocity
        self.file_path_def = os.path.join(self.log_path, f"def_weight_{self.seed}.csv")
        with open(self.file_path_def, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Weights"])

    def _on_step(self) -> bool:
        
        #info: Training
        # Get the reward of the current step
        reward = self.locals["reward"][0]

        # Update the current episode reward
        self.cumm_reward += reward

        #info: Check if the episode is done
        if self.locals.get("done"):
            # Append the episode reward to the list
            self.episode_rewards.append(self.cumm_reward)
            
            self.logger.record("rew_last100eps_avg",
                               np.mean(self.episode_rewards[-100:]))
            
            if np.mean(self.episode_rewards[-100:]) > self.max_r_till_now:
                self.max_r_till_now = np.mean(self.episode_rewards[-100:])

            
            if self.num_eps % 10 == 0:
                # every 10 eps print avg reward of last 100 eps
                avg_reward = np.mean(self.episode_rewards[-100:])
                logger.info(f"(Train) Episode: {self.num_eps}, Avg Reward: {avg_reward}, Best: {self.max_r_till_now}")
                
            
            self.num_eps += 1
            self.cumm_reward = 0 # reset cumm reward

            ## evaluate for 100 episodes at each 100th episode
            if self.num_eps % 100 == 0:
                test_rewards, test_lengths = evaluate_policy(
                    self.model,
                    self.eval_env,
                    n_eval_episodes=100,
                    render=self.render,
                    deterministic=self.deterministic,
                    return_episode_rewards=True,
                    warn=self.warn,
                    callback=self._log_success_callback,
                )
                
                test_mean_reward, test_std_reward = np.mean(test_rewards), np.std(test_rewards)
                min_test_reward = np.min(test_rewards)
                self.test_rewards.append(test_mean_reward) # append to the list the mean of 100 episodes
                self.worst_rewards.append(min_test_reward)
                logger.info(f"(Test) Episode: {100}, Avg Test Reward: {test_mean_reward}, Min Test Rewards: {min_test_reward}")
                
                
                

        #info: Evaluating
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            sync_envs_normalization(self.training_env, self.eval_env)

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            #infor: since log_path is none (not saving), this part is not executed
            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                ## info: nothing is saved here
                # np.savez(
                #     self.log_path,
                #     timesteps=self.evaluations_timesteps,
                #     results=self.evaluations_results,
                #     ep_lengths=self.evaluations_length,
                #     **kwargs,
                # )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward
            
            if self.verbose > 0:
                logger.info(f"(Eval ) num_timesteps={self.num_timesteps}, Reward={mean_reward:.2f} +/- {std_reward:.2f}, EpLength: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            #what: What is this part doing?
            #info: this part is not executed since log_path is none
            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    logger.info("(Save ) Storing Best Model: New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, f"best_model_seed{self.seed}"))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()
            # SPP: Saves a lot of checkpoints name_timesteps
            # info: No need save checkpoint for now (consumes memory)
            # if mean_reward > self.minimum_reward:
            #     if self.best_model_save_path is not None:
            #         self.model.save(os.path.join(self.best_model_save_path, f"seed{self.seed}/callback_{self.n_calls}"))

        return True
    


