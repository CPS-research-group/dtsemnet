"""Generate Test results for stored agents
The agents are trained using the code from ProLoNet codebase"""

import numpy as np
import os
from monitor.eval_agent import test_agent,test_sc_agent
from agents.dtnet.agent import DTNetAgent
from agents.fcnn.agent import FCNNAgent
from agents.prolonet.agent import DeepProLoNet
import gym
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = ""

AGENT_TYPE = 'dtnet'  # 'dtnet', 'fcnn', 'prolo'
ENV_TYPE = 'cart'  # 'lunar', 'cart', 'acrobot'$
DT_TYPE = 'l1'
RANDOM = False
USE_GPU = False
PATH = ''
DATE = datetime.datetime.now().strftime("%Y-%m-%d") # append to file name
MODEL_PATH = 'trained_models/cart_gpu/dtnet/l1/adam_nodup'
# MODEL_PATH = 'trained_models/cart_gpu/dtnet/l1/adam_dup_rand'
# MODEL_PATH = 'trained_models/cart_gpu/prolo/fixed/'
# MODEL_PATH = 'trained_models/cart_gpu/prolo/fixed_rand/'
# MODEL_PATH = 'trained_models/lunar_gpu/prolo/fixed_nodup_31N/'
# MODEL_PATH = 'trained_models/lunar_gpu/prolo/63N_deep/'
# MODEL_PATH = 'trained_models/lunar_gpu/prolo/fixed_nodup_63N_rand/'


# MODEL_PATH = 'trained_models/FindAndDefeatZerglings_gpu/prolo/fixed_10n_rand/'

# MODEL_PATH = 'trained_models/FindAndDefeatZerglings_gpu/dtnet/v1/rms_rn_rand/'

#MODEL_PATH = 'trained_models/FindAndDefeatZerglings_gpu/prolo/fixed_8N/'
# MODEL_PATH = 'trained_models/FindAndDefeatZerglings_gpu/prolo/fixed_8N_rand/'

# MODEL_PATH = 'trained_models/lunar_gpu/dtnet/minimal/rms_rand/'

if ENV_TYPE == 'lunar':
    env = gym.make('LunarLander-v2')
    dim_in = 8
    dim_out = 4
    NN_NODES = 31
elif ENV_TYPE == 'cart':
    env = gym.make('CartPole-v1')
    dim_in = 4
    dim_out = 2
elif ENV_TYPE == 'zerlings':
    dim_in = 37
    dim_out = 10
    NN_NODES = '10n'


# Define Agent
if AGENT_TYPE == 'dtnet':
    print(f'===========> Testing DTNet Agent')
    print(f'===========> Model Loaded from {MODEL_PATH}')
    policy_agent = DTNetAgent(  env_name=ENV_TYPE,
                                path=PATH,
                                dt_type=DT_TYPE,
                                use_gpu=USE_GPU,
                                input_dim=dim_in,
                                output_dim=dim_out,
                                random=RANDOM,
                                deterministic=True,
                                num_dt_nodes=NN_NODES)
elif AGENT_TYPE == 'prolo':
    print(f'============> Test ProloNet Agent')
    print(f'===========> Model Loaded from {MODEL_PATH}')
    policy_agent = DeepProLoNet(distribution='one_hot',
                                path=PATH,
                                input_dim=dim_in,
                                output_dim=dim_out,
                                use_gpu=USE_GPU,
                                vectorized=False,
                                randomized=False,
                                adversarial=False,
                                deepen=False,
                                deterministic=True)
elif AGENT_TYPE == 'fcnn':
    print(f'============> Test FCNN Agent {MODEL_PATH}')
    policy_agent = FCNNAgent(env_name=ENV_TYPE,
                                path=PATH,
                                input_dim=dim_in,
                                output_dim=dim_out,
                                use_gpu=USE_GPU,
                                deterministic=True)





for i in range(5):
    policy_agent.load(act_fn=f'{MODEL_PATH}/seed{i}_actor.tar')
    test_reward_array = np.array([])
    for seed in range(500, 600):
        # ========== Seed for Test Envs [500, 501 .... 599] ==========
        # Test agent for 1eps with seed
        if ENV_TYPE == 'zerlings':
            reward = test_sc_agent(seed=seed,
                            policy_agent=policy_agent)
        else:
            reward = test_agent(env=env,
                                seed=seed,
                                policy_agent=policy_agent,
                                test_dt=False,
                                deterministic=True) # prolonet agents are not deterministic
        test_reward_array = np.append(test_reward_array, reward)
    # calculate mean reward over 100 episodes
    test_reward_mean = np.mean(test_reward_array)
    test_reward_std = np.std(test_reward_array)
    print(f'=====> Agent {i} Test Reward: {test_reward_mean}')
    test_rew_fp = f'results/{ENV_TYPE}_{"gpu" if USE_GPU else "cpu"}/{AGENT_TYPE}{"_rand" if RANDOM else ""}_rms_test_rewards_{DATE}.txt'
    with open(test_rew_fp, 'a') as test_rew_file:
        test_rew_file.write(f"\n{test_reward_mean:.2f} +/- {test_reward_std}")