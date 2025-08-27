import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dprl.agents.fcnn import FCNN
from dprl.agents.dtsemnet_topk import DTSemNet as dtnet_topk
from dprl.agents.dgt import DGT

class SoftQNetwork(nn.Module):
    def __init__(self, env, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            hidden_dim,
        )
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class fcnn_actor(nn.Module):
    def __init__(self, env, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 64)
        self.fc2 = nn.Linear(64,64)
        self.fc_mean = nn.Linear(64, np.prod(env.single_action_space.shape))
        # self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Parameter(torch.zeros(1), requires_grad=True)
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        # log_std = self.fc_logstd(x)
        log_std = self.fc_logstd
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x, deterministic=False):
        mean, log_std = self(x)
        std = log_std.exp()
        
        if deterministic:
            y_t = torch.tanh(mean)
            action = y_t * self.action_scale + self.action_bias
            log_prob = None  # not needed for evaluation
        else:
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # reparameterization trick
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

class dtsemnet_topk_actor(nn.Module):
    def __init__(self, env, **kwargs):
        super().__init__()
        # self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        # self.fc2 = nn.Linear(256, 256)
        self.mode = 'test'
        self.fc_mean = dtnet_topk(
                            in_dim=np.array(env.single_observation_space.shape).prod(),
                            out_dim=np.prod(env.single_action_space.shape),
                            height=kwargs.get('height', 4),  
                            is_regression=True,
                            over_param=[],
                            linear_control=True,
                            wt_init='none',
                            reg_hidden=0,
                            ste = "hard_argmax_ste",
                            top_k= kwargs.get('top_k', 4),  # 4 for training, 1 for evaluation
                            detach=False,
                            smax_temp= kwargs.get('smax_temp', 1.0),  # 0.5 for training, 1.0 for evaluation
                            batch_norm=False,
                        )
        # training: (top4, 0.5), (top2, 1.0)
        # have single learnable parameter for logstd not inedependent of the input
        # self.fc_logstd = nn.Parameter(torch.zeros(1), requires_grad=True)
        # self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        # self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        
        
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
    
    def set_mode(self, mode: str):
        assert mode in ['topk', 'aug', 'top1'], f"Unsupported mode: {mode}"
        if mode == 'topk':
            self.mode = 'train'
        elif mode == 'aug':
            self.mode = 'aug'
        elif mode == 'top1':
            self.mode = 'test'
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'topk', 'aug', or 'top1'.")
        return self

    def forward(self, x):
        mean, log_std = self.fc_mean(x, mode=self.mode)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        

        return mean, log_std

    def get_action(self, x, deterministic=False):
        mean, log_std = self(x)
        std = log_std.exp()
        # print('mean', mean)
        # print('std', std)
        # exit()
        
        if deterministic:
            y_t = torch.tanh(mean)
            action = y_t * self.action_scale + self.action_bias
            log_prob = None  # not needed for evaluation
        else:
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # reparameterization trick
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


def agent_loader(agent):
    if agent == 'fcnn':
        actor_class = fcnn_actor
        # critic = FCNN(input_dim=kwargs['input_dim'], hidden_layers=kwargs['critic_arch'], output_dim=kwargs['output_dim'], policy=False)
    elif agent == 'dtsemnet_topk':
        actor_class = dtsemnet_topk_actor
        # critic = dtnet_topk(in_dim=kwargs['input_dim'], out_dim=kwargs['output_dim'], height=kwargs['depth'], is_regression=True, over_param=[], linear_control=True, wt_init='none', reg_hidden=0, ste="hard_argmax_ste", top_k=4, detach=False, smax_temp=0.5)
        
    
    # Q-Network architecture is same for all agents
    critic_class = SoftQNetwork
    
    return actor_class, critic_class


# Example usage:
# fcnn_agent = agent_loader('FCNN', input_size=10, hidden_layers=[20, 20], output_size=1)
# dtsemnet_agent = agent_loader('DTSemNet', input_size=10, depth=5, output_size=1)