"""Training script for various agent.
Implementation reused from the GitHub code of ProLoNet. It raises some warnings.
Usage: python train_agent.py -a [agent_type] -e [episodes] -p [processes] -env [env_type] -gpu -vec -adv -rand -deep -s -reproduce

Various Discrete Action Envs
Gym:
CartPole-v1
Acrobot-v1
LunarLander-v2

Others:
FindAndDefeatZerglings
"""

import copy
import argparse
from learner.replay_buffer import discount_reward
import random

from agents.dtnet.agent import DTNetAgent
from agents.prolonet.agent import DeepProLoNet
from agents.fcnn.agent import FCNNAgent
from monitor.eval_agent import test_agent
import gym
import numpy as np

import torch
import os
import datetime

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import logging



def run_episode(q, agent_in, env, env_seed_in):
    # SPP: 1a. Duplicate agent?
    # SPP: We need to dublicate because the replay buffer is extended in the main function, so the entries will be duplicated
    #
    if type(agent_in).__name__ == 'DTNetAgent':
        if agent_in.env_name == 'cart' and agent_in.random == True:
            agent = agent_in.duplicate()
        else:
            agent = agent_in # controls learning rate (makes it twice the lr) indirectly
    else:
        if agent_in.env_name == 'lunar': # or agent_in.env_name == 'acrobot':
            agent = agent_in
        else:
            agent = agent_in.duplicate() # for prolonet

    agent.action_network.eval()

    env.seed(env_seed_in)
    np.random.seed(env_seed_in)
    torch.random.manual_seed(env_seed_in)
    random.seed(env_seed_in)

    state = env.reset()  # Reset environment and record the starting state
    done = False

    while not done:
        action = agent.get_action(state)
        # Step through environment using chosen action
        state, reward, done, _ = env.step(action)
        # Save reward
        # SPP: 1b. Save state, action and reward in replay buffer
        agent.buffer_insert(state, action, reward)
        if done:
            break
    reward_sum = np.sum(agent.replay_buffer.rewards_list) # SPP: Sum of rewards of the current episode

    # SPP: what is discount_reward?
    rewards_list, advantage_list, deeper_advantage_list = discount_reward(
        agent.replay_buffer.rewards_list, agent.replay_buffer.value_list,
        agent.replay_buffer.deeper_value_list)
    # SPP: 1c. Update replay buffer
    agent.replay_buffer.rewards_list = rewards_list
    agent.replay_buffer.advantage_list = advantage_list
    agent.replay_buffer.deeper_advantage_list = deeper_advantage_list

    to_return = [reward_sum, copy.deepcopy(agent.replay_buffer.__getstate__())]
    if q is not None:
        try:
            q.put(to_return)
        except RuntimeError as e:
            print(e)
            return to_return
    return to_return


def main(episodes, agent, init_env, random_seed, ix=0):
    """Function to train agent.
    1. Run the episode till it ends (done=True)
    """
    running_reward_array = [] # stores reward of each episode
    max_reward = -np.inf  # used to save the best model
    for episode in range(episodes):
        if random_seed is not None:
            random_seed_in = random_seed + episode
        else:
            random_seed_in = random_seed

        # SPP: Agent is DTNet and following activities are performed
        # SPP: 1. run_episode
        returned_object = run_episode(None,
                                      agent_in=agent,
                                      env=init_env,
                                      env_seed_in=random_seed_in)
        ep_reward = returned_object[0]/float(1) # SPP: Reward from the current Episode (converted to float)
        # print(ep_reward)
        running_reward_array.append(ep_reward)
        # SPP: 2. Extend replay buffer: Why clearing the replay buffer and then extending it?
        agent.replay_buffer.extend(returned_object[1]) # SPP: It's actually UPDATING the replay buffer not EXTENDING it

        # SPP: 3. Update agent
        # SPP: This stores reward of each episode in txt file
        # SPP: It also resets the replay buffer to empty
        agent.action_network.train()
        agent.value_network.train()
        agent.end_episode()

        running_reward = sum(running_reward_array[-100:]) / float(min(100.0, len(running_reward_array)))

        if episode % 50 == 0:
            logging.info(
                f'Episode {episode}  Current Episode Reward: {ep_reward}  Average (last 100) Running Reward: {np.round(running_reward, 2)}'
            )

        # SPP: Save the model if if return is highest
        if episode%5 == 0:
            # evaluate env on seed 20
            # test_reward = test_agent(init_env, 20, agent, test_dt=False)
            if running_reward > max_reward:
                max_reward = running_reward
                policy_agent.save(ix=ix)  # SPP: 5. Save model
                logging.info(
                    f'Model saved for seed {ix}. Max Test reward: {max_reward}'
                )

        # SPP: 4. Lower learning rate at every 250th episode
        if episode % 250 == 0:
            agent.lower_lr()




    # running_reward_array: stores reward of num_episodes(1000) [ep_rew1, ep_rew2, ep_rew3, ...]
    # running_reward: reward of the last 100 episodes
    return running_reward_array, running_reward, max_reward


# SPP: Calls "main" function with cart-env + prolo + 1 process
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a",
                        "--agent_type",
                        help="Agent Architecture",
                        type=str,
                        default='dtnet')
    parser.add_argument("-dt",
                        "--dt_type",
                        help="DT Architecture",
                        type=str)
    parser.add_argument("-nn", "--num_nodes", help="DT Internal Nodes", type=int, default=31)
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
    parser.add_argument("-deep", help="Deepen prolonet?", action='store_true')
    parser.add_argument("-rand", help="Random Weight Initialization", action='store_true')
    parser.add_argument("--reproduce",
                        help="use saved random seeds and make deterministic?",
                        action="store_true")

    args = parser.parse_args()
    AGENT_TYPE = args.agent_type  # 'dtnet'
    NUM_EPS = args.episodes  # num episodes Default 1000
    ENV_TYPE = args.env_type  # 'cart' or 'lunar' Default 'cart'
    USE_GPU = args.gpu  # Default false
    DT_TYPE = args.dt_type  # 'l1' or 'l2' or 'minimal' Default 'l1'
    NN_NODES = args.num_nodes  # Default 31, number of decision nodes in tree. Default 31 is for lunar lander
    # Cart: 11
    # Zerglings: 8
    
    DEEPEN = args.deep  # Default false
    EPSILON = 1.0  # Chance to take random action
    EPSILON_DECAY = 0.99  # Multiplicative decay on epsilon
    EPSILON_MIN = 0.1  # Floor for epsilon
    RANDOM = args.rand  # Default false

    REPRODUCE = args.reproduce
    torch.use_deterministic_algorithms(True)


    log_spec = input(f'Log Spec: ')
    user_input = input(f"Extra Info on this training: ")
    if AGENT_TYPE == 'dtnet':
        assert DT_TYPE in ['l1', 'l2', 'minimal', 'v1'
                           ], "DT_TYPE must be one of 'l1', 'l2', 'minimal', 'v1'"
        PATH = f"{ENV_TYPE}_{'gpu' if USE_GPU else 'cpu'}/{AGENT_TYPE}/{DT_TYPE}/{log_spec}/"
    else:
        PATH = f"{ENV_TYPE}_{'gpu' if USE_GPU else 'cpu'}/{AGENT_TYPE}/{log_spec}/"

    log_fp = f"logs/{PATH}"
    if not os.path.exists(log_fp):
        os.makedirs(log_fp)



    # Logging path
    logging.basicConfig(level=logging.INFO,
                        handlers=[
                            logging.FileHandler(f'{log_fp}/training.log',
                                                mode='w'),
                            logging.StreamHandler()
                        ])

    if ENV_TYPE == 'lunar':
        init_env = gym.make('LunarLander-v2')
    elif ENV_TYPE == 'cart':
        init_env = gym.make('CartPole-v1')
    elif ENV_TYPE == 'mcar':
        init_env = gym.make('MountainCar-v0')
        init_env._max_episode_steps = 1000
    # elif ENV_TYPE == 'invdblpend': # inverted double pendulum
    #     init_env = gym.make('InvertedDoublePendulum-v4') # ?: Continuous action space
    elif ENV_TYPE == 'acrobot': # acrobot
        init_env = gym.make('Acrobot-v1')
    # elif ENV_TYPE == 'bipedal':
    #     init_env = gym.make('BipedalWalker-v3') # ?: Continuous action space
    else:
        raise Exception('No valid environment selected')

    dim_in = init_env.observation_space.shape[0]
    dim_out = init_env.action_space.n

    if REPRODUCE:
        if ENV_TYPE == 'cart':
            seeds = [100, 62, 65, 104, 112]
        elif ENV_TYPE == 'lunar':
            seeds = [42, 46, 48, 50, 51]
        elif ENV_TYPE == 'acrobot':
            seeds = [100, 105, 110, 115, 120]
        else:
            seeds = [45, 48, 49, 53, 67]




    logging.info(f"Training {AGENT_TYPE} on {ENV_TYPE} for {NUM_EPS} episodes.")
    logging.info(f"Description: {user_input}")
    logging.info(f"DT Type: {DT_TYPE}")
    if USE_GPU:
        logging.info(f'Using GPU: {USE_GPU}')

    logging.info(f'PATH: {PATH}')
    logging.info(f'Random: {RANDOM}')
    logging.info(f'Training seed: {seeds}')
    logging.info(f'State dim: {dim_in} Action dim: {dim_out}')
    logging.info(f'Train Params: discount: {0.99} epsilon: {EPSILON}, epsilon_decay: {EPSILON_DECAY}, epsilon_min: {EPSILON_MIN}')

    # Generate a timestamp string in the format of "YYYY-MM-DD HH:MM:SS"
    timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Append the model training entry to the log file
    with open('on_going_training.md', 'a') as log_file:
        log_file.write(f"\n| {timestamp_str} | {user_input} | {PATH} |")

    # xxxxxxxxxxxxxxxxxxxxx Training Starts xxxxxxx
    for i in range(5):  # SPP: Experiment repeated 5 times
        if REPRODUCE:
            # SPP: Make training deterministic
            seed_in = seeds[i]
            init_env.seed(seed_in)
            np.random.seed(seed_in)
            torch.random.manual_seed(seed_in)
            random.seed(seed_in)
            os.environ['PYTHONHASHSEED'] = str(seed_in)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = False
            torch.cuda.manual_seed_all(seed_in)
            torch.cuda.manual_seed(seed_in)
        else:
            seed_in = None

        if AGENT_TYPE == 'dtnet':
            logging.info(f'Training DTNet Agent for seed: {i}')
            policy_agent = DTNetAgent(  env_name=ENV_TYPE,
                                        path=PATH,
                                        dt_type=DT_TYPE,
                                        input_dim=dim_in,
                                        output_dim=dim_out,
                                        use_gpu=USE_GPU,
                                        epsilon=EPSILON,
                                        epsilon_decay=EPSILON_DECAY,
                                        epsilon_min=EPSILON_MIN,
                                        random=RANDOM,
                                        deterministic=False,
                                        num_dt_nodes=NN_NODES,)
        elif AGENT_TYPE == 'prolo':
            logging.info(f'Training ProLoNet Agent for seed: {i}')
            policy_agent = DeepProLoNet(distribution='one_hot',
                                        path=PATH,
                                        input_dim=dim_in,
                                        output_dim=dim_out,
                                        use_gpu=USE_GPU,
                                        vectorized=False,
                                        randomized=RANDOM,
                                        adversarial=False,
                                        deepen=DEEPEN,
                                        epsilon=EPSILON,
                                        epsilon_decay=EPSILON_DECAY,
                                        epsilon_min=EPSILON_MIN,
                                        deterministic=False)

        elif AGENT_TYPE == 'fcnn':
            logging.info(f'Training FCNN Agent for seed: {i}')
            policy_agent = FCNNAgent(env_name=ENV_TYPE,
                                     path=PATH,
                                     input_dim=dim_in,
                                     output_dim=dim_out,
                                     use_gpu=USE_GPU,
                                     epsilon=EPSILON,
                                     epsilon_decay=EPSILON_DECAY,
                                     epsilon_min=EPSILON_MIN,
                                     deterministic=False)
        else:
            raise Exception('No valid agent selected')

        # TODO: Fix these
        logging.info(f'>>>>>>>>>>>>> Training for seed: {i} <<<<<<<<<<<<<<<<<<<')
        reward_array, avg_running_reward, max_reward = main(NUM_EPS, policy_agent, init_env, seed_in, ix=i)

        # Store rewards for each seed in txt file
        reward_fp = f'{log_fp}/seed{str(i)}_rewards.txt'
        with open(reward_fp, 'w') as myfile:
            np.savetxt(myfile, reward_array, fmt='%.1f')

        # xxxxxxxxxxxxxxxxxxxxx Training Ends; Test the model xxxxxxx
        # TODO: Test the trained model and store results in txt file for each seed
        # Test the trained model and store results in txt file for each seed
        if AGENT_TYPE == 'dtnet' or AGENT_TYPE == 'fcnn':
            policy_agent.load(act_fn=f'trained_models/{PATH}seed{i}_actor.tar')

        if AGENT_TYPE == 'prolo':
            if DEEPEN:
                policy_agent.load(
                    act_fn=f'trained_models/{PATH}seed{i}_actor.tar',
                    deep_act_fn=f'trained_models/{PATH}seed{i}_deep_actor.tar')
            else:
                policy_agent.load(
                    act_fn=f'trained_models/{PATH}seed{i}_actor.tar')
        # seed 10 is unseen test environment
        ep_test_rew = test_agent(init_env, 10, policy_agent, test_dt=False)
        # Store test results in txt file (deterministic=True)
        logging.info(f'=====> Test reward for seed {i}: {ep_test_rew} <=====')
        if not os.path.exists(f'results/{PATH}'):
            os.makedirs(f'results/{PATH}')
        test_rew_fp = f'results/{PATH}test_rewards_{datetime.datetime.now().strftime("%Y-%m-%d")}.txt'
        with open(test_rew_fp, 'a') as test_rew_file:
            test_rew_file.write(f"\n{ep_test_rew}")
