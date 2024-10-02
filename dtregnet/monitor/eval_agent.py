"""Various functions to test the agent at the end of training
test_single_episode: test agent in environment for a single episode
running_avg: calculate running average of rewards over 100 episodes
"""

import numpy as np
import random
import torch
import gym

# from sc2.constants import *
# from sc2.position import Pointlike, Point2
# from sc2.player import Bot, Computer
# from sc2.unit import Unit as sc2Unit
# import sc2
# from sc2 import Race, Difficulty
import numpy as np



# from sc2.constants import *
# from sc2.position import Pointlike, Point2
# from sc2.player import Bot, Computer
# from sc2.unit import Unit as sc2Unit


# from utils import sc_helpers
import numpy as np


DEBUG = False
SUPER_DEBUG = False
if SUPER_DEBUG:
    DEBUG = True

FAILED_REWARD = -0.0
SUCCESS_BUILD_REWARD = 1.
SUCCESS_TRAIN_REWARD = 1.
SUCCESS_SCOUT_REWARD = 1.
SUCCESS_ATTACK_REWARD = 1.
SUCCESS_MINING_REWARD = 1.


def test_agent(env, seed, policy_agent, test_dt=False, deterministic=True):
    """Test agent at the end of training and store it's results.

    Arguments:
        seed {int} -- seed for reproducibility
        policy_agent {DTNetAgent} -- agent to test
    
    Keyword Arguments:
        test_dt {bool} -- whether to test the DT or the agent (default: {False})
    
    Returns:
        float -- total reward for the episode
    """
    # policy_agent.deterministic = True  # set agent to deterministic
    policy_agent.action_network.eval()  # set action network to evaluation mode
    policy_agent.deterministic = deterministic
    env.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)

    state = env.reset()  # Reset environment and record the starting state
    done = False
    total_reward = 0
    ret_obs = None
    while not done:
        # prolonet action
        obs = torch.Tensor(state)
        obs = obs.view(1, -1)
        action = policy_agent.get_action(obs)

        if test_dt:
            # SPP:only applicable for DTNetAgent
            action = policy_agent.dt.get_action(obs)
            action = int(action.item())
            if action != policy_agent.get_action(obs):
                print(obs, action, policy_agent.get_action(obs))
                ret_obs = obs
        else:
            action = policy_agent.get_action(obs)


        # Step through environment using chosen action
        state, reward, done, _ = env.step(action)  # Gym env step
        # add reward to total reward

        total_reward += reward
        if done:
            break

    policy_agent.deterministic = False  # set agent to non-deterministic
    policy_agent.action_network.train()  # set action network to training mode
    if test_dt:
        return total_reward, ret_obs
    else:
        return total_reward


# class SC2MicroBot(sc2.BotAI):
#     def __init__(self, rl_agent, kill_reward=1):
#         super(SC2MicroBot, self).__init__()
#         self.agent = rl_agent
#         self.kill_reward = kill_reward
#         self.action_buffer = []
#         self.prev_state = None
#         self.last_known_enemy_units = []
#         self.itercount = 0
#         self.last_reward = 0
#         self.my_tags = None
#         self.agent_list = []
#         self.dead_agent_list = []
#         self.dead_index_mover = 0
#         self.dead_enemies = 0

#     async def on_step(self, iteration):

#         if iteration == 0:
#             self.my_tags = [unit.tag for unit in self.units]
#             for unit in self.units:
#                 self.agent_list.append(self.agent.duplicate())
#         else:
#             self.last_reward = 0
#             for unit in self.state.dead_units:
#                 if unit in self.my_tags:
#                     self.last_reward -= 1
#                     self.dead_agent_list.append(
#                         self.agent_list[self.my_tags.index(unit)])
#                     del self.agent_list[self.my_tags.index(unit)]
#                     del self.my_tags[self.my_tags.index(unit)]
#                     self.dead_agent_list[-1].buffer_insert(
#                         reward=self.last_reward)
#                 else:
#                     self.last_reward += self.kill_reward
#                     self.dead_enemies += 1
#             # if len(self.state.dead_units) > 0:
#             for agent in self.agent_list:
#                 agent.buffer_insert(reward=self.last_reward)
#         for unit in self.units:
#             if unit.tag not in self.my_tags:
#                 self.my_tags.append(unit.tag)
#                 self.agent_list.append(self.agent.duplicate())
#         # if iteration % 20 != 0:
#         #     return
#         all_unit_data = []
#         for unit in self.units:
#             all_unit_data.append(sc_helpers.get_unit_data(unit))
#         while len(all_unit_data) < 3:
#             all_unit_data.append([-1, -1, -1, -1])
#         for unit, agent in zip(self.units, self.agent_list):
#             nearest_allies = sc_helpers.get_nearest_enemies(unit, self.units)
#             unit_data = sc_helpers.get_unit_data(unit)
#             nearest_enemies = sc_helpers.get_nearest_enemies(
#                 unit, self.known_enemy_units)
#             unit_data = np.array(unit_data).reshape(-1)
#             enemy_data = []
#             allied_data = []
#             for enemy in nearest_enemies:
#                 enemy_data.extend(sc_helpers.get_enemy_unit_data(enemy))
#             for ally in nearest_allies[1:3]:
#                 allied_data.extend(sc_helpers.get_unit_data(ally))
#             enemy_data = np.array(enemy_data).reshape(-1)
#             allied_data = np.array(allied_data).reshape(-1)
#             state_in = np.concatenate((unit_data, allied_data, enemy_data))
#             action = agent.get_action(state_in)
#             await self.execute_unit_action(unit, action, nearest_enemies)
#         try:
#             await self.do_actions(self.action_buffer)
#         except sc2.protocol.ProtocolError:
#             print("Not in game?")
#             self.action_buffer = []
#             return
#         self.action_buffer = []

#     async def execute_unit_action(self, unit_in, action_in, nearest_enemies):
#         if action_in < 4:
#             await self.move_unit(unit_in, action_in)
#         elif action_in < 9:
#             await self.attack_nearest(unit_in, action_in, nearest_enemies)
#         else:
#             pass

#     async def move_unit(self, unit_to_move, direction):
#         current_pos = unit_to_move.position
#         target_destination = current_pos
#         if direction == 0:
#             target_destination = [current_pos.x, current_pos.y + 5]
#         elif direction == 1:
#             target_destination = [current_pos.x + 5, current_pos.y]
#         elif direction == 2:
#             target_destination = [current_pos.x, current_pos.y - 5]
#         elif direction == 3:
#             target_destination = [current_pos.x - 5, current_pos.y]
#         self.action_buffer.append(
#             unit_to_move.move(Point2(Pointlike(target_destination))))

#     async def attack_nearest(self, unit_to_attack, action_in,
#                              nearest_enemies_list):
#         if len(nearest_enemies_list) > action_in - 4:
#             target = nearest_enemies_list[action_in - 4]
#             if target is None:
#                 return -1
#             self.action_buffer.append(unit_to_attack.attack(target))
#         else:
#             return -1

#     def finish_episode(self, game_result):
#         print("Game over!")
#         if game_result == sc2.Result.Defeat:
#             for index in range(len(self.agent_list), 0, -1):
#                 self.dead_agent_list.append(self.agent_list[index - 1])
#                 self.dead_agent_list[-1].buffer_insert(reward=-1)
#             del self.agent_list[:]
#         elif game_result == sc2.Result.Tie:
#             reward = 0
#         elif game_result == sc2.Result.Victory:
#             reward = 0  # - min(self.itercount/500.0, 900) + self.units.amount
#         else:
#             # ???
#             return -13
#         if len(self.agent_list) > 0:
#             reward_sum = sum(self.agent_list[0].replay_buffer.rewards_list)
#         else:
#             reward_sum = sum(
#                 self.dead_agent_list[-1].replay_buffer.rewards_list)

#         for agent_index in range(len(self.agent_list)):
#             rewards_list, advantage_list, deeper_advantage_list = discount_reward(
#                 self.agent_list[agent_index].replay_buffer.rewards_list,
#                 self.agent_list[agent_index].replay_buffer.value_list,
#                 self.agent_list[agent_index].replay_buffer.deeper_value_list)
#             self.agent_list[
#                 agent_index].replay_buffer.rewards_list = rewards_list
#             self.agent_list[
#                 agent_index].replay_buffer.advantage_list = advantage_list
#             self.agent_list[
#                 agent_index].replay_buffer.deeper_advantage_list = deeper_advantage_list
#         for dead_agent_index in range(len(self.dead_agent_list)):
#             rewards_list, advantage_list, deeper_advantage_list = discount_reward(
#                 self.dead_agent_list[dead_agent_index].replay_buffer.
#                 rewards_list, self.dead_agent_list[dead_agent_index].
#                 replay_buffer.value_list,
#                 self.dead_agent_list[dead_agent_index].replay_buffer.
#                 deeper_value_list)
#             self.dead_agent_list[
#                 dead_agent_index].replay_buffer.rewards_list = rewards_list
#             self.dead_agent_list[
#                 dead_agent_index].replay_buffer.advantage_list = advantage_list
#             self.dead_agent_list[
#                 dead_agent_index].replay_buffer.deeper_advantage_list = deeper_advantage_list
#         return self.dead_enemies * self.kill_reward - len(self.dead_agent_list)


def run_episode(q, main_agent, game_mode):
    result = None
    agent_in = main_agent
    agent_in.action_network.eval()
    kill_reward = 1
    if game_mode == 'DefeatRoaches':
        kill_reward = 10
    elif game_mode == 'DefeatZerglingsAndBanelings':
        kill_reward = 5
    bot = SC2MicroBot(rl_agent=agent_in, kill_reward=kill_reward)

    try:
        result = sc2.run_game(
            sc2.maps.get(game_mode),
            [Bot(Race.Terran, bot)],
            realtime=False,
        )
    except KeyboardInterrupt:
        result = [-1, -1]
    except Exception as e:
        print(str(e))
        print("No worries", e, " carry on please")
    if type(result) == list and len(result) > 1:
        result = result[0]
    reward_sum = bot.finish_episode(result)
    for agent in bot.agent_list + bot.dead_agent_list:
        agent_in.replay_buffer.extend(agent.replay_buffer.__getstate__())
    if q is not None:
        try:
            q.put([reward_sum, agent_in.replay_buffer.__getstate__()])
        except RuntimeError as e:
            print(e)
            return [reward_sum, agent_in.replay_buffer.__getstate__()]
    return reward_sum #sends episode reward


def test_sc_agent(seed, policy_agent):
    """Test agent at the end of training and store it's results.

    Arguments:
        policy_agent {DTNetAgent} -- agent to test
    
    
    Returns:
        float -- total reward for the episode
    """
    # policy_agent.deterministic = True  # set agent to deterministic
    policy_agent.action_network.eval()  # set action network to evaluation mode
    policy_agent.deterministic = True
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


    total_reward = run_episode(None,
                                  main_agent=policy_agent,
                                  game_mode='FindAndDefeatZerglings')

    policy_agent.deterministic = False
    policy_agent.action_network.train()  # set action network to training mode

    return total_reward


def running_avg(arr, num_seeds=5, N=100):
    """Calculate running average of rewards over 100 episodes
    
    Arguments:
        arr {list} -- list of rewards for each seed
        num_seeds {int} -- number of seeds

    Returns:
        np.array -- array of running average rewards for each seed
    """
    run_avg_arr = []
    for s in range(num_seeds):
        run_avg = [arr[s][0]]
        for i in range(len(arr[s]) - 1):
            if i < N: # for previous 100 steps
                ix = i + 1
                ra = sum(arr[s][0:ix]) / float(len(arr[s][0:ix]))

            else:
                ix = i + 1
                ra = sum(arr[s][ix-int(N):ix]) / N

            # print(ra)
            run_avg.append(ra)
        run_avg_arr.append(run_avg)

    run_avg_arr = np.array(run_avg_arr)
    return run_avg_arr


def get_confidence_interval(arr):
    """Calculate confidence interval for array of rewards.
    
    Arguments:
        arr {np.array} -- array of rewards
    
    Returns:
        tuple -- mean, lower bound, upper bound
    """
    mean = np.mean(arr, axis=0)
    std = np.std(arr, ddof=1, axis=0) / np.sqrt(arr.shape[0])
    cil, cih = mean - std, mean + std
    return mean, cil, cih
