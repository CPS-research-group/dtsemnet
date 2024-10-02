"""Policies: abstract base class and concrete implementations.
Used implementations from stable-baselines3 package.
Modified the actor-critic policy agent to custom architecture
"""

import collections
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
import torch

from stable_baselines3.common.type_aliases import Schedule

from stable_baselines3.common.policies import BasePolicy
from agents.dt.agent import DTAgent
from agents.fcnn.arch import FCNN
from agents.dgt.arch import DGT
from torch.distributions import Categorical


import logging


from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_device




"""Actor Critic Agent of Discrete Action Space"""""
class ACAgent(BasePolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[
            BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        env_name: str = 'cart',
        rand: bool = False,
        agent_name: str = 'dtnet',
        num_dt_nodes: int = 31,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        self.rand = rand
        self.env_name = env_name
        self.agent_name = agent_name
        self.num_dt_nodes = num_dt_nodes
        self.log_std_init = log_std_init
        self.ortho_init = ortho_init
        device = get_device()  # get available device

        if self.agent_name == 'dtnet':
            print('Using DTNet')
            if self.env_name == 'cart':
                self.dt = DTAgent(env='cart')  # use l1 DT
            elif self.env_name in ['lunar', 'acrobot']:
                self.dt = DTAgent(
                    env=self.env_name,
                    dt_type='minimal',
                    num_dt_nodes=self.num_dt_nodes)  # use minimal DT
            elif self.env_name in ['zerlings']:
                self.dt = DTAgent(env=self.env_name, dt_type='v1')

            # Get action and value network from DT
            # Model is loaded on the GPU if available
            self.action_net, self.value_net = self.dt.get_DTNet(
                env=self.env_name,
                random=self.rand,
                use_gpu=device == torch.device('cuda'),
                val_arch = None,
                sb=True)

        elif self.agent_name == 'fcnn':
            print('Using FCNN')
            if self.env_name == 'cart':
                self.input_dim = 4
                self.output_dim = 2
                self.action_net = FCNN(
                    self.input_dim,
                    self.output_dim,
                    env=self.env_name,
                    action_net=True,
                )
            elif self.env_name == 'lunar':
                self.input_dim = 8
                self.output_dim = 4
                # Action Net Lunar: 64x64 and cart 32x32
                self.action_net = FCNN(self.input_dim,
                                       self.output_dim,
                                       env=self.env_name,
                                       action_net=True,
                                       arch='64x64')
            elif self.env_name == 'acrobot':
                self.input_dim = 6
                self.output_dim = 3
                self.action_net = FCNN(self.input_dim,
                                       self.output_dim,
                                       env=self.env_name,
                                       action_net=True,
                                       arch='64x64')

            #Value Net Lunar: 64x64x32 and cart 32x32, zerglings: 256x256x126
            self.value_net = FCNN(
                self.input_dim,
                self.output_dim,
                env=self.env_name,
                action_net=False,
                value_net_single_op=True,  # single output for value net single
            )



            # Move networks to GPU if needed
            if device == torch.device('cuda'):
                self.action_net = self.action_net.cuda()
                self.value_net = self.value_net.cuda()
        
        # =========== DGT ===========
        elif self.agent_name == 'dgt':
            print('Using DGT')
            if self.env_name == 'cart':
                self.input_dim = 4
                self.output_dim = 2
                # for height of 4 with 15 internal nodes
                height = 6
                logging.info('Using DGT with height: {}'.format(height))
                self.action_net = DGT(4,2, height=height)

            elif self.env_name == 'lunar':
                self.input_dim = 8
                self.output_dim = 4
                # for height of 6 with 63 internal nodes
                self.action_net = DGT(8,4, height=6)

            elif self.env_name == 'acrobot':
                self.input_dim = 6
                self.output_dim = 3
                # for height of 4 with 15 internal nodes
                self.action_net = DGT(6,3, height=5)

        # =========== Value Net ===========
            self.value_net = nn.Sequential(nn.Linear(self.input_dim, 64),
                                           nn.ReLU(), nn.Linear(64, 64),
                                           nn.ReLU(), nn.Linear(64, 1))

            for module in self.value_net.modules():
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, np.sqrt(2))
                    nn.init.zeros_(module.bias)

            # Move networks to GPU if needed
            if device == torch.device('cuda'):
                self.action_net = self.action_net.cuda()
                self.value_net = self.value_net.cuda()



        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(),
                                              lr=lr_schedule(1),
                                              **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(
            lambda: None)

        data.update(
            dict(
                log_std_init=self.log_std_init,
                ortho_init=self.ortho_init,
                use_sde=self.use_sde,
                full_std=self.full_std,
                use_expln=self.use_expln,
                squash_output=self.squash_output,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                share_features_extractor=self.share_features_extractor,
                normalize_images=self.normalize_images,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                rand=self.rand,
                env_name=self.env_name,
                agent_name=self.agent_name,
            ))
        return data

    def forward(
            self,
            obs: th.Tensor,
            deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """

        # Evaluate the values for the given observations
        values = self.value_net(obs)
        probs = self.action_net(obs)
        distribution = Categorical(probs=probs)

        if deterministic:
            actions = torch.argmax(distribution.probs, dim=1)
        else:
            actions = distribution.sample()

        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def _predict(self,
                 observation: th.Tensor,
                 deterministic: bool = False) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        probs = self.action_net(observation)
        distribution = Categorical(probs=probs)

        if deterministic:
            actions = torch.argmax(distribution.probs, dim=1)
        else:
            actions = distribution.sample()

        return actions

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed

        values = self.value_net(obs)
        probs = self.action_net(obs)
        distribution = Categorical(probs=probs)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(obs)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        return self.value_net(obs)


