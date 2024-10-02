"""Decision tree agent. Builds decision tree and converts it to DTNet
Contains all the decision tree architectures.
"""

import numpy as np
import torch
from agents.dt.arch.cart import DT_cart
from agents.dt.arch.lunar import DT_lunar_minimal, DT_lunar, DT_lunar2, DT_lunar_v1
from agents.dt.arch.mcar import DT_mcar_v1
from agents.dt.arch.acrobot import DT_acrobot_minimal
from agents.dt.arch.zerlings import DT_zerlings_v1, DT_zerlings_l2, DT_zerlings_minimal
from agents.dtnet.arch import DTNetv1, DTNetv0
from agents.fcnn.arch import FCNN
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)




class DTAgent:
    """Agent that uses a decision tree to make decisions"""
    def __init__(self,
                 env,
                 state_dim=None,
                 action_dim=None,
                 num_decision_nodes=None,
                 num_action_nodes=None,
                 node_weights=None,
                 node_bias=None,
                 init_leaves=None,
                 dt_type=None,
                 num_dt_nodes=None,):

        if env is None:
            raise ValueError('Environment is not defined')
        self.env = env
        if state_dim != None:
            self.state_dim = state_dim
        if action_dim != None:
            self.action_dim = action_dim
        if node_weights != None:
            self.W = node_weights
        if node_bias != None:
            self.B = node_bias
        if init_leaves != None:
            self.init_leaves = init_leaves
        self.dt_type = dt_type

        self.num_dt_nodes = num_dt_nodes

        if num_decision_nodes != None:
            self.num_decision_nodes = num_decision_nodes
        if num_action_nodes != None:
            self.num_action_nodes = num_action_nodes

        ########

        # l1: Authors' implementation
        # minimal: complete binary
        # v1: decision and action nodes are separate
        ########

        if self.env == 'cart': # Create a decision tree for the cartpole environment
            self.state_dim, self.action_dim, self.W, self.B, self.init_leaves = DT_cart(
            )  # initialize cartpole DT

        elif self.env == 'lunar': # Create a decision tree for the lunar lander environment
            # lunar2: Leaves updated from Prolonet
            if self.dt_type == 'l1':
                logger.debug('Creating lunar DT l1')
                self.state_dim, self.action_dim, self.W, self.B, self.init_leaves = DT_lunar()
            elif self.dt_type == 'l2':
                logger.debug('Creating lunar DT l2')
                self.state_dim, self.action_dim, self.W, self.B, self.init_leaves = DT_lunar2() # initialize lunar lander DT
            elif self.dt_type == 'minimal':
                logger.debug('Creating lunar DT minimal')

                self.state_dim, self.action_dim, self.W, self.B, self.init_leaves = DT_lunar_minimal(nodes=self.num_dt_nodes
                )
            elif self.dt_type == 'v1':
                logger.debug('Creating lunar DT v1')
                self.state_dim, self.action_dim, self.num_decision_nodes, self.num_action_nodes, self.W, self.B, self.init_leaves = DT_lunar_v1(
                )

        elif self.env == 'mcar': # Create a decision tree for the mountain car environment
            if self.dt_type == 'v1':
                logger.debug('Creating mcar DT v1')
                self.state_dim, self.action_dim, self.num_decision_nodes, self.num_action_nodes, self.W, self.B, self.init_leaves = DT_mcar_v1(
                )

        elif self.env == 'acrobot':  # Create a decision tree for the mountain car environment
            if self.dt_type == 'v1':
                logger.debug('Creating acrobot DT v1')
                self.state_dim, self.action_dim, self.num_decision_nodes, self.num_action_nodes, self.W, self.B, self.init_leaves = DT_acrobot_v1(
                )
            elif self.dt_type == 'minimal':
                logger.debug('Creating acrobot DT minimal')
                self.state_dim, self.action_dim, self.W, self.B, self.init_leaves = DT_acrobot_minimal(
                )

        elif self.env == 'zerlings':
            if self.dt_type == 'v1':
                logger.debug('Creating zerlings DT v1')
                self.state_dim, self.action_dim, self.num_decision_nodes, self.num_action_nodes, self.W, self.B, self.init_leaves = DT_zerlings_v1(num_nodes=self.num_dt_nodes
                )
            elif self.dt_type == 'l2':
                logger.debug('Creating lunar DT l2')
                self.state_dim, self.action_dim, self.W, self.B, self.init_leaves = DT_zerlings_l2(
                )
            elif self.dt_type == 'minimal':
                logger.debug('Creating lunar DT minimal')
                self.state_dim, self.action_dim, self.W, self.B, self.init_leaves = DT_zerlings_minimal(
                )

        

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

    def get_DTNet(self, use_gpu, env=None, random=False, sb=False, val_arch=None):
        """Returns the DTNet model as action network and FCNN as value network"""
        
        if self.dt_type == 'v1':
            action_network = DTNetv1(
                input_dim=self.state_dim,
                output_dim=self.action_dim,
                node_weights=self.W,
                node_bias=self.B,
                leaf_weights=self.L,
                leaf_actions=self.A,
                init_leaves=self.init_leaves,
                num_decision_nodes=self.num_decision_nodes,
                num_action_nodes=self.num_action_nodes,
                action_net=True,
                random=random,
            )
        
        else:
            action_network = DTNetv0(input_dim=self.state_dim,
                                     output_dim=self.action_dim,
                                     node_weights=self.W,
                                     node_bias=self.B,
                                     leaf_weights=self.L,
                                     leaf_actions=self.A,
                                     init_leaves=self.init_leaves,
                                     random=random,)

        value_network = FCNN(input_dim=self.state_dim,
                             output_dim=self.action_dim,
                             env=env,
                             action_net=False,
                             value_net_single_op=sb,
                             arch=val_arch
                             )
        if use_gpu:
            action_network = action_network.cuda()
            value_network = value_network.cuda()
        return action_network, value_network

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
        # print(f'Leaves Weights: {self.L}')
        # print(f'Leaf Actions: {self.A}')
        return "DTAgent"

    def __str__(self) -> str:
        return f"DT Agent for {self.env} environment"
