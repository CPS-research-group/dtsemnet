# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
import datetime
import json
import logging
import argparse

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace
import stable_baselines3 as sb3

from dprl.agents.agent_loader import agent_loader



# ALGO LOGIC: initialize agent here:



def evaluate_policy(actor, env_id, n_eval_episodes=30, deterministic=True):
    episode_rewards = []

    for i in range(n_eval_episodes):
        env = make_env(env_id, seed=500 + i, idx=i, capture_video=False, run_name="eval")()
        obs, _ = env.reset(seed=100 + i)
        total_reward = 0.0
        done = False
        

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                actions, _, _ = actor.get_action(obs_tensor, deterministic=deterministic)
            action = actions.cpu().numpy()[0]
            
            next_obs, rewards, terminations, truncations, _ = env.step(action)
            total_reward += rewards
            obs = next_obs
            done = terminations or truncations

        episode_rewards.append(total_reward)
        env.close()

    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    return avg_reward, std_reward


def make_env(env_id, seed, idx, capture_video, run_name):
    if env_id == "lunar":
        env_name = "LunarLanderContinuous-v2"
    elif env_id == "walker":
        env_name = "BipedalWalker-v3"
    else:
        raise ValueError(f"Unsupported environment id: {env_id}")
    
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_name, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"logs/videos/{run_name}")
        else:
            env = gym.make(env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    
    return thunk

def parse_args():
    """
    Parse the command-line arguments and load the config file.
    Returns the parsed arguments as a dictionary.
    """
    parser = argparse.ArgumentParser(description="Train a model.")
    
    # Add a command-line argument for the config file
    parser.add_argument('--config', type=str, required=True, help="Path to the config JSON file.")
    parser.add_argument('--seed', type=int, required=False, help="Seed for reproducibility")
    parser.add_argument('--exp_name', type=str, required=False, help="Name of the log directory")
    

    
    # Parse the arguments
    args = parser.parse_args()
    
    # Load the JSON config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # replace config from command line if provided
    if args.seed:
        config["seed"] = args.seed
    if args.exp_name:
        config["exp_name"] = args.exp_name
    
    # Convert the config dictionary to a SimpleNamespace for easier attribute access
    config = SimpleNamespace(**config)

    # Return all the parameters
    return config

def log_config(args):
    """
    Log the configuration parameters.
    """
    # name appended to log directory
    # generate logname based on time

    log_dir = f'{args.env_id}_{"gpu" if args.cuda else "cpu"}/{args.agent}/{args.exp_name}/'
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
            
    return log_dir


if __name__ == "__main__":
    
    ############### Initialize the parameters ###############
    #########################################################
    # Parse arguments
    args = parse_args()
    
    # disable GPU visibility if not using GPU
    if not args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # check user inputs
    assert args.agent in ['dtsemnet_topk', 'dtsemnet_ste', 'fcnn'], "Invalid Agent Type"
    assert args.env_id in ['lunar', 'walker'], "Invalid Environment Type"
    
    seed = args.seed
    actor_arch = args.actor_arch
    critic_arch = args.critic_arch
    
    
    log_dir = log_config(args)
    timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') # Generate a timestamp string in the format of "YYYY-MM-DD HH:MM:SS"

    ############### Log Basic Information ###############
    #########################################################
    # Configure Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s[%(name)s]%(levelname)s | %(message)s',
        datefmt='%d-%m-%y %H:%M',
        handlers=[
            logging.FileHandler(f'./logs/{log_dir}training{seed}.log', mode='w'),
            logging.StreamHandler()
        ])
    logging.info(f"Training on {args.env_id} environment for {args.episodes} episodes")
    logging.info(f"Agent Type: {args.agent}" )
    logging.info(f"Log Directory: {log_dir}")
    logging.info(f"Random Weight Initialization: {args.wt_init}")
    logging.info(f"Training Started: {timestamp_str}")
    json_str = json.dumps(vars(args))  # Convert args to a JSON string
    logging.info('Hyperparameters: %s', json_str) # Log the JSON string
    logging.info(f"Actor Architecture: {actor_arch}")
    logging.info(f"Critic Architecture: {critic_arch}")
    logging.info(f"STE: {args.ste}")

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"logs/{log_dir}/runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    
    ############### Seeding ###############
    #########################################################
    # TRY NOT TO MODIFY: (seeding)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    os.environ['PYTHONHASHSEED'] = str(seed)


    

    ############### Environment Setup ###############
    #########################################################
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    logging.info(f"Created environment {args.env_id}")
    
    ############### Initialize Networks and Optimizer #######
    #########################################################
    max_action = float(envs.single_action_space.high[0])

    # laod the agent
    # Modules are just network architectures
    ActorClass, CriticClass = agent_loader(agent=args.agent)
    if args.agent in ['dtsemnet_topk', 'dtsemnet_ste']:
        actor = ActorClass(envs, height=args.height, topk=args.topk, smax_temp=args.smax_temp).to(device)
    else:
        actor = ActorClass(envs).to(device)
    qf1 = CriticClass(envs, hidden_dim=args.critic_arch).to(device)
    qf2 = CriticClass(envs, hidden_dim=args.critic_arch).to(device)
    qf1_target = CriticClass(envs, hidden_dim=args.critic_arch).to(device)
    qf2_target = CriticClass(envs, hidden_dim=args.critic_arch).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32

    print(f"Actor: {actor}")
    
    ## Initialize Replay Buffer
    #########################################################
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    
    ## TRAINING STARTS HERE
    logging.info(f"===================> Training Started with seed {seed} <=================") # log the seed
    start_time = time.time()
    episodic_reward = []
    episodic_step = []
    best_test_reward = -np.inf  # Initialize best test reward for evaluation
    episode_num = 0

    

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    episodic_reward.append(info["episode"]["r"])
                    episodic_step.append(info["episode"]["l"])
                    episode_num = len(episodic_reward)
                    # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    if episode_num % 10 == 0: # train log each 10 episodes
                        logging.info(f"Episode: {episode_num}, Step: {episodic_step[-1]} Avg Reward (last 100eps): {np.mean(episodic_reward[-100:]):.2f}")
                    
                    ## Evaluate the policy every eval_freq episodes
                    #############################
                    
                    if episode_num % args.eval_freq == 0 and global_step > args.learning_starts:  # evaluate every eval_freq episodes
                        if args.agent in ['dtsemnet_topk']:   
                            actor.set_mode('top1')  # Set actor mode to test (top-1)
                        avg_reward, std_reward = evaluate_policy(actor, args.env_id, n_eval_episodes=30, deterministic=True)
                        logging.info(f"==>Evaluation at episode {episode_num} step {global_step}: Avg Reward (ep{args.eval_freq}): {avg_reward:.2f} ± {std_reward:.2f}")
                        writer.add_scalar("charts/eval/avg_reward", avg_reward, global_step)
                        writer.add_scalar("charts/eval/std_reward", std_reward, global_step)
                        # save the model
                        if avg_reward > best_test_reward:
                            best_test_reward = avg_reward
                            logging.info(f"==>New best test reward: {best_test_reward:.2f}")
                            # save the model with seed and log_dir
                            if episode_num > 1000: # save only after 500 episodes
                                logging.info(f"-->Saving best model<--")
                                model_path_best = f"trained_models/{log_dir}seed{seed}_best_model.pth"
                                torch.save(actor.state_dict(), model_path_best)

                    
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        #################
        if global_step > args.learning_starts and global_step % args.train_freq == 0:

            # Training
            #################
            # For Top-k 
            if args.agent in ['dtsemnet_topk']:   
                actor.set_mode("topk")  # Set actor mode to train with topk
                for param_group in actor_optimizer.param_groups:
                    param_group['lr'] = args.policy_lr
            for _ in range(args.gradient_steps):
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                    qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = qf1(data.observations, data.actions).view(-1)
                qf2_a_values = qf2(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # update the actor
                pi, log_pi, _ = actor.get_action(data.observations)
                qf1_pi = qf1(data.observations, pi)
                qf2_pi = qf2(data.observations, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(data.observations)
                    alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            # if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # train with Top-1 (only for top1)
            #################
            if args.agent in ['dtsemnet_topk']: 
                # train with Top-1 (Augmented Samples)
                #################
                actor.set_mode('aug')  # Set actor mode to test (top-1)
                for param_group in actor_optimizer.param_groups:
                    param_group['lr'] = args.top1_lr
                for _ in range(args.gradient_steps):
                    data = rb.sample(args.batch_size)
                    pi, log_pi, _ = actor.get_action(data.observations)
                    
                    aug_obs = data.observations.repeat_interleave(2, dim=0)  # [2B, obs_dim]
                    
                    # Step 3: Q-function evaluations on 2B observations and actions
                    qf1_pi = qf1(aug_obs, pi)  # [2B, 1]
                    qf2_pi = qf2(aug_obs, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)

                    # Step 4: Actor loss
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                
                
                # train with Top-1 Final training
                #################
                actor.set_mode('top1') # Set actor mode to test (top-1)
                for param_group in actor_optimizer.param_groups:
                    param_group['lr'] = args.top1_lr
                for _ in range(args.gradient_steps//2):
                    data = rb.sample(args.batch_size)

                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

            
            ## Record
            #################
            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                # print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        if episode_num > args.episodes:
            logging.info(f"Training completed after {episode_num} episodes and {global_step} steps.")
            break
    ############### Save Network and Episodic Step and Reward ###############
    if args.agent in ['dtsemnet_topk']:   
        actor.set_mode('top1')  # Set actor mode to test (top-1)
    logging.info(f"===================> Storing Reward <=================")
    # # Save the rewards
    rew = np.array(episodic_reward) #only first 1000 episodes
    reward_fp = f'logs/{log_dir}'
    fn = f"{reward_fp}seed{seed}_train_rewards.txt"
    with open(fn, "w") as file:
        np.savetxt(fn, rew, fmt='%.1f')
    # Save the agent
    # todo: save the model in a more structured way
    model_path_last = f"trained_models/{log_dir}seed{seed}_last_model.pth"
    torch.save(actor.state_dict(), model_path_last)
    


    ############### Evaluate the Agent ###############
    #########################################################
    actor.eval()
    if args.agent in ['dtsemnet_topk']:   
        actor.set_mode('top1')  # Set actor mode to test (top-1)
    logging.info(f"===================> Evaluation Started (Last Model-Reported for Publication) <=================")
    test_reward_avg, test_reward_std = evaluate_policy(actor, args.env_id, n_eval_episodes=100, deterministic=True)
    # Store test results in txt file (deterministic=True)
    logging.info(f'==============> Test reward for seeds [500, 600]: {test_reward_avg:.2f} +/- {test_reward_std} <==========')


    logging.info(f"===================> Evaluation Started (Best Model) <=================")
    # Load the best model
    actor.load_state_dict(torch.load(model_path_best))
    actor.eval()
    if args.agent in ['dtsemnet_topk']:   
        actor.set_mode('top1')  # Set actor mode to test (top-1)
    test_reward_avg, test_reward_std = evaluate_policy(actor, args.env_id, n_eval_episodes=100, deterministic=True)
    # Store test results in txt file (deterministic=True)
    logging.info(f'==============> Test reward for seeds [500, 600]: {test_reward_avg:.2f} +/- {test_reward_std} <==========')

    envs.close()
    writer.close()