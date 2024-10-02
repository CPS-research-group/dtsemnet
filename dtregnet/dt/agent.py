"""Decision tree agent. Builds decision tree and converts it to DTNet
Contains all the decision tree architectures.
"""

import numpy as np
import torch

from dtregnet.dt.trees.completeDT import genDT
from dtregnet.agent.arch import DTSemNetReg

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)




class DTAgent:
    """Agent that uses a decision tree to make decisions"""
    def __init__(self,
                 env,
                 state_dim=None,
                 action_dim=None,
                 num_leaf=None,
                 lin_control=True,
                 hk=False):

        if env is None:
            raise ValueError('Environment is not defined')
        
        self.env = env
        self.num_leaf = num_leaf
        self.state_dim = state_dim
        self.action_dim = action_dim # not the actual action dimension, but the number of leaf nodes
        self.lin_control = lin_control
        self.hk = hk

        
        #info: Get Complete Binary Tree for Lunar Lander environment
        if self.env in ['cart', 'lunar', 'lane_keeping', 'ring_accel', 'highway', 'intersection', 'racetrack', 'walker', 'ugrid']:
            self.state_dim, self.action_dim, self.W, self.B, self.init_leaves, self.num_controls = genDT(env_name=self.env, num_leaf=self.num_leaf)
        else:
            raise NotImplementedError

        self.L, self.A = self.leaves_to_weight(self.init_leaves, self.W.shape[0])


    def get_action(self, x):
        """Get action from decision tree given state
        Args:
            x: state of form [[i, j, k, l]] where i, j, k, l are floats. Shape: (num_samples, num_features)
        """
        comp = np.matmul(x, self.W.T) + self.B
        num_inputs, _ = x.shape

        actions = torch.zeros(num_inputs)
        for i in range(num_inputs):
            a = 0
            for leaf in self.init_leaves:
                if torch.all(comp[i][leaf[0]] >= 0) and torch.all(
                        comp[i][leaf[1]] < 0):
                    actions[i] = np.argmax(
                        leaf[-1])  # take argmax of leaf action

                    a += 1
                    org_l = leaf.copy()
                    if a > 1:
                        print('leaf1:', org_l)
                        print('leaf2:', leaf)
                        raise ValueError('More than one leaf satisfied')


        return actions


    def leaves_to_weight(self, init_leaves, num_nodes):
        """
        Converts the leaves of the decision tree to the weights of the DTNet.
        :param init_leaves: list of leaves of the decision tree
        :param num_nodes: number of nodes in the decision tree
        :return: Matrix of weights of condition layer in DTNet (layer 3)
        """
        num_leaves = len(init_leaves)
        L = np.zeros((num_leaves, 2 * num_nodes))  # each row represents a leaf
        A = np.zeros(num_leaves)  # action associated with each leaf
        for ix, leaf in enumerate(init_leaves):
            if len(leaf[0]) > 0:  # True Conditions
                # For Relu(D) for each node
                L[ix][leaf[0]] = 1
            if len(leaf[1]) > 0:  # False Conditions
                # For Relu(-D) for each node
                L[ix][[nx + num_nodes for nx in leaf[1]]] = 1

            # don't care conditions: need to but D+ and D- for each node
            dont_cares = set(range(num_nodes)) - set(
                leaf[0] + leaf[1])  # don't care conditions
            if len(dont_cares) > 0:
                L[ix][list(dont_cares)] = 1
                L[ix][[nx + num_nodes for nx in dont_cares]] = 1

            A[ix] = np.argmax(
                leaf[-1])  # action associated with leaf 0: Left and 1: Right

        return L, A

    def get_DTRegNet(self):
        """Returns DTSemNetRegresion model"""
        action_network = DTSemNetReg(
            input_dim=self.state_dim,
            output_dim=self.action_dim,
            node_weights=self.W,
            node_bias=self.B,
            leaf_weights=self.L,
            leaf_actions=self.A,
            init_leaves=self.init_leaves,
            lin_control=self.lin_control,
            num_controls=self.num_controls,
            hk=self.hk
        )
        return action_network
        

    def update_weights(self, node_weights, node_bias):
        """Update the weights of the decision tree"""
        self.W = node_weights
        self.B = node_bias

    def __repr__(self) -> str:
        """Prints the decision tree agent"""
        print(f'Agent: Decision Tree')
        print(f'Environment: {self.env}')
        print(f'State dimension: {self.state_dim}')
        print(f'Action Dimension: {self.action_dim}')
        print(f'Number of nodes: {self.W.shape[0]}')
        print(f'Number of leaves: {len(self.init_leaves)}')
        print(f'Node weights: {self.W}')
        print(f'Node biases: {self.B}')
        print(f'Leaf nodes: {self.num_leaf}')
        print(f'Linear control: {self.lin_control}')
        print(f'Number of controls: {self.num_controls}')
        return "DTAgent"

    def __str__(self) -> str:
        return f"DT Agent for {self.env} environment"
