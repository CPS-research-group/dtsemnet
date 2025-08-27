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
from torch.distributions import Categorical

from dprl.agents.agent_loader import agent_loader

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

        env_name: str = None,
        rand: bool = False,
        agent_type: str = None,
        num_leaves: int = None,
        actor_arch: list = None,
        critic_arch: list = None,
        ste: str = None,
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
        self.agent_type = agent_type
        self.num_leaves = num_leaves
        self.log_std_init = log_std_init
        self.ortho_init = ortho_init
        self.actor_arch = actor_arch
        self.critic_arch = critic_arch
        self.ste = ste
        device = get_device()  # get available device

        self.input_dim = observation_space.shape[0]
        self.output_dim = action_space.n

        ## initialize the agent
        # call agent loader

        self.actor, self.critic = agent_loader(agent_type=self.agent_type,
                                                       input_dim=self.input_dim,
                                                       output_dim=self.output_dim,
                                                       actor_arch=self.actor_arch,
                                                       critic_arch=self.critic_arch,
                                                       num_leaves=self.num_leaves,
                                                       rand=self.rand,
                                                       ste=self.ste)

        # Move networks to GPU if needed
        if device == torch.device('cuda'):
            self.actor = self.actor.cuda()
            self.critic = self.critic.cuda()

        
        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(),
                                              lr=lr_schedule(1),
                                              **self.optimizer_kwargs)
        # log the network architecture

        logging.info('Action Net Arch: %s', self.actor.__str__())
        logging.info('Value Net Arch: %s', self.critic.__str__())
        

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
                agent_type=self.agent_type,
                num_leaves=self.num_leaves,
                actor_arch=self.actor_arch,
                critic_arch=self.critic_arch,

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
        values = self.critic(obs)
        probs = self.actor(obs)
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
        probs = self.actor(observation)
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

        values = self.critic(obs)
        probs = self.actor(obs)
        distribution = Categorical(probs=probs)
        log_prob = distribution.log_prob(actions)
        values = self.critic(obs)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        return self.critic(obs)
