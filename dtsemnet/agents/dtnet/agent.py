'''DTNet Agent
Agent Structure is similar to ProLoNets work.

ProLoNets: 
https://github.com/CORE-Robotics-Lab/ProLoNets
'''

import torch
from torch.distributions import Categorical
from learner import replay_buffer, ppo_update
from agents.dt.agent import DTAgent
import os
import logging
import numpy as np

class DTNetAgent:
    """Agent that uses a DTNet to select actions."""
    def __init__(self,
                 env_name,
                 path,
                 dt_type='l1',
                 input_dim=4,
                 output_dim=2,
                 use_gpu=True,
                 epsilon=0.9,
                 epsilon_decay=0.95,
                 epsilon_min=0.05,
                 random=False,
                 deterministic=False,
                 num_dt_nodes=31,):
        self.replay_buffer = replay_buffer.ReplayBufferSingleAgent()
        self.env_name = env_name # name of the environemnt
        self.path = path # name of the log file and type of execution
        self.dt_type = dt_type
        self.use_gpu = use_gpu
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.adv_prob = .05
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.deterministic = deterministic
        self.lr = 2e-2
        self.random = random

        self.num_dt_nodes = num_dt_nodes  # Applicable only for lunar lander
        

        # Initialize DT for the environment
        if self.env_name == 'cart':
            self.dt = DTAgent(env='cart')
        elif self.env_name == 'pend':
            self.dt = DTAgent(env='pend')
        elif self.env_name in ['lunar', 'mcar', 'acrobot', 'zerlings']:
            self.dt = DTAgent(env=self.env_name, dt_type=self.dt_type, num_dt_nodes=self.num_dt_nodes)

        # Get action and value network
        self.action_network, self.value_network = self.dt.get_DTNet(use_gpu, self.env_name, random=random, sb=False)

        # Initialize PPO: Training Algorithm
        # SPP: Commented out PPO for testing Pendulum
        # self.ppo = ppo_update.PPO([self.action_network, self.value_network], two_nets=True, use_gpu=use_gpu, agent='dtnet', random=self.random)
        # self.actor_opt = torch.optim.Adam(self.action_network.parameters(), lr=1e-5)
        # self.value_opt = torch.optim.RMSprop(self.value_network.parameters(), lr=1e-5)


        self.last_state = [0, 0, 0, 0]
        self.last_action = 0
        self.last_action_probs = torch.Tensor([0])
        self.last_value_pred = torch.Tensor([[0, 0]])
        self.full_probs = None
        self.reward_history = []
        self.num_steps = 0

    def get_action(self, observation):
        """Returns an action given an observation."""
        with torch.no_grad():
            obs = torch.Tensor(observation)
            obs = obs.view(1, -1)
            self.last_state = obs
            if self.use_gpu:
                obs = obs.cuda()

            probs = self.action_network(obs)
            value_pred = self.value_network(obs)
            probs = probs.view(-1).cpu()
            self.full_probs = probs
            if self.action_network.input_dim >= 33:
                probs, inds = torch.topk(probs, 3)
            m = Categorical(probs)
            action = m.sample()
            if self.deterministic:
                action = torch.argmax(probs)

            log_probs = m.log_prob(action)
            self.last_action_probs = log_probs.cpu()
            self.last_value_pred = value_pred.view(-1).cpu()


            if self.action_network.input_dim >= 33:
                self.last_action = inds[action].cpu()
            else:
                self.last_action = action.cpu()
        if self.action_network.input_dim >= 33:
            action = inds[action].item()
        else:
            action = action.item()
        return action

    def buffer_insert(self, state=None, action=None, reward=None):
        """Insert state action and reward tupple to buffer."""
        if self.deterministic:
            return True
        self.replay_buffer.insert(obs=[self.last_state],
                                  action_log_probs=self.last_action_probs,
                                  value_preds=self.last_value_pred[self.last_action.item()],
                                  last_action=self.last_action,
                                  full_probs_vector=self.full_probs,
                                  rewards=reward)
        return True

    def end_episode(self):
        """End episode and update actor and critic networks."""
        self.action_network.train()
        self.value_network.train()
        value_loss, action_loss = self.ppo.batch_updates(self.replay_buffer, self)
        # SPP: PPO Update resets the replay buffer
        self.num_steps += 1
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)

    def lower_lr(self):
        """Lower learning rate by factor of 2."""
        # no decay in learning rate
        if self.env_name == 'zerlings':
            rate = 0.8
        else:
            rate = 0.5 # for lunar and cart
        self.lr = self.lr * rate
        for param_group in self.ppo.actor_opt.param_groups:
            param_group['lr'] = param_group['lr'] * rate
        for param_group in self.ppo.critic_opt.param_groups:
            param_group['lr'] = param_group['lr'] * rate

    def reset(self):
        """Reset replay buffer."""
        self.replay_buffer.clear()

    def save(self, fn='trained_models/', ix=0):
        """Save actor to file."""
        act_fn = f"{fn}{self.path}"
        if not os.path.exists(act_fn):
            os.makedirs(act_fn)
        act_fn += f"seed{ix}_actor.tar"

        self.action_network.save_dtnet(act_fn)
        print('Saved actor to: ' + act_fn)

    def load(self, act_fn=None):
        """Load actor from file."""
        if os.path.exists(act_fn):
            self.action_network = self.action_network.load_dtnet(act_fn)
            if self.use_gpu:
                self.action_network = self.action_network.cuda()
            self.dt = self.latest_DTAgent() # update DTAgent
            print('Loaded actor from: ' + act_fn)
        else:
            raise Exception('Actor file not found: ' + act_fn)


    def latest_DTAgent(self):
        """Let Latest DT Agent from the DTNet Agent."""
        if self.dt_type == 'v1':
            W1 = self.action_network.linear_decision.weight.data.detach().clone().data.cpu().numpy()
            W2 = self.action_network.linear_action.weight.data.detach().clone().data.cpu().numpy()
            B1 = self.action_network.linear_decision.bias.data.detach().clone().data.cpu().numpy()
            B2 = self.action_network.linear_action.bias.data.detach().clone().data.cpu().numpy()
            W = np.concatenate((W1, W2), axis=0)
            B = np.concatenate((B1, B2), axis=0)
        else:
            W = self.action_network.linear1.weight.data.detach().clone().data.cpu().numpy()
            B = self.action_network.linear1.bias.data.detach().clone().data.cpu().numpy()

        self.dt.update_weights(node_weights = W, node_bias = B)
        return self.dt

    def __getstate__(self):
        return {
            'action_network': self.action_network,
            'value_network': self.value_network,
            'ppo': self.ppo,
            'actor_opt': self.ppo.actor_opt,
            'critic_opt': self.ppo.critic_opt,
            'path': self.path,
            'use_gpu': self.use_gpu,
            'output_dim': self.output_dim,
            'input_dim': self.input_dim,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'deterministic': self.deterministic,
            'random': self.random,
            'env_name': self.env_name,
            'dt_type': self.dt_type,
            'num_dt_nodes': self.num_dt_nodes,
        }

    def __setstate__(self, state):
        for key in state:
            setattr(self, key, state[key])

    def deepen_networks(self):
        return

    def duplicate(self):
        """Return a duplicate of the agent."""
        new_agent = DTNetAgent(env_name=self.env_name,
                               path=self.path,
                               dt_type=self.dt_type,
                               input_dim=self.input_dim,
                               output_dim=self.output_dim,
                               use_gpu=self.use_gpu,
                               epsilon=self.epsilon,
                               epsilon_decay=self.epsilon_decay,
                               epsilon_min=self.epsilon_min,
                               random=self.random,
                               deterministic=self.deterministic,
                               num_dt_nodes=self.num_dt_nodes,
                               )
        new_agent.__setstate__(self.__getstate__())
        return new_agent
