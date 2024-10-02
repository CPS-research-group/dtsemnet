''' 
This file is the actual contribution of this work.

DTNet Architecture for the Decision Tree Network.
Layer 1: Linear Layer
Layer 2: Relu
Layer 3: Linear Layer (w/o) activation to add various conditions
Layer 4: MaxPool Layer
Layer 5: Softmax
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import typing as t


class MaxPoolLayer(nn.Module):
    """Custom MaxPool Layer for DTNet."""
    def __init__(self, leaf_actions):
        super(MaxPoolLayer, self).__init__()
        leaf_actions = np.array(leaf_actions, dtype='object')
        actions = np.unique(leaf_actions)
        self.node_groups = []
        for action in actions:
            self.node_groups.append(np.where(leaf_actions == action)[0])

        self.num_groups = len(self.node_groups)  #first 2 conditions


    def forward(self, x):
        batch_size, _ = x.size()
        # initialize output tensor with zeros
        out = torch.zeros((batch_size, self.num_groups)).to(x.device)

        # loop over node groups and compute the maximum value for each group
        for i, group in enumerate(self.node_groups):
            group_nodes = x[:, group]
            max_values, _ = torch.max(group_nodes, dim=1)
            out[:, i] = max_values

        return out


class DTNetv1(nn.Module):
    """DTNet Architecture for the Decision Tree Network."""
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 node_weights: np.ndarray,
                 node_bias: np.ndarray,
                 leaf_weights: np.ndarray,
                 leaf_actions: np.ndarray,
                 init_leaves: t.List,
                 num_decision_nodes: int = None,
                 num_action_nodes: int = None,
                 action_net: bool = True,
                 random: bool = False):
        super(DTNetv1, self).__init__()

        self.in_features = input_dim  # input dimension of state
        self.input_dim = input_dim  # input dimension of state # SPP: refactor this
        self.output_dim = output_dim
        self.num_decision_nodes = num_decision_nodes
        self.num_action_nodes = num_action_nodes
        self.action_net = action_net
        self.random = random
        assert self.num_decision_nodes + self.num_action_nodes + 1 == len(
            leaf_actions
        ), "Number of internal nodes in DT should be one less than the number of leaves"

        self.num_leaves = len(leaf_actions)  # number of leaves in DT
        self.init_leaves = init_leaves  # Leaf Nodes as list: Used to getback the DT from DTNet

        # Layer1: Linear Layer
        self.linear_decision = nn.Linear(self.in_features,
                                         self.num_decision_nodes)
        self.linear_action = nn.Linear(self.in_features, self.num_action_nodes)

        self.init_weights_d = torch.tensor(
            node_weights[0:self.num_decision_nodes], dtype=torch.float32)
        self.init_weights_a = torch.tensor(
            node_weights[self.num_decision_nodes:], dtype=torch.float32)
        self.init_biases_d = torch.tensor(node_bias[0:self.num_decision_nodes],
                                          dtype=torch.float32)
        self.init_biases_a = torch.tensor(node_bias[self.num_decision_nodes:],
                                          dtype=torch.float32)

        self.linear_decision.weight.data = self.init_weights_d  # intialize weights from DT weights
        self.linear_action.weight.data = self.init_weights_a  # intialize weights from DT weights
        self.linear_action.weight.requires_grad = False  # Fixed weights (no gradient)

        self.linear_decision.bias.data = self.init_biases_d  # initialize weights from DT bias
        self.linear_action.bias.data = self.init_biases_a  # initialize weights from DT bias

        # RANDOM INITIALIZATION
        if self.random:


            # Orhtogonal Initialization of Weights
            nn.init.orthogonal_(self.linear_decision.weight)

            # Uniform Initialization of Biases
            if self.input_dim <= 8:  # cart and lunar
                nn.init.uniform_(self.linear_decision.bias, a=-1, b=1)
                nn.init.uniform_(self.linear_action.bias, a=-1, b=1)
            else:  # for zerglings
                nn.init.uniform_(self.linear_decision.bias, a=-40, b=40)
                nn.init.uniform_(self.linear_action.bias, a=-40, b=40)
            # linear_action.weight is fixed (no gradient)

        # Layer2: Relu Activation for Relu(+D) and Relu(-D)
        self.reluP = nn.ReLU()
        self.reluM = nn.ReLU()

        # Layer3: Linear Layer (w/o) activation to add various conditions
        self.linear_leaf = nn.Linear(
            2 * (self.num_action_nodes + self.num_decision_nodes),
            self.num_leaves,
            bias=False)
        self.init_weights_leaf = torch.tensor(leaf_weights,
                                              dtype=torch.float32)
        self.linear_leaf.weight.data = self.init_weights_leaf  # intialize weights from DT
        self.linear_leaf.weight.requires_grad = False  # Fixed weights (no gradient)

        # Layer4: MaxPool Layer
        self.leaf_actions = leaf_actions
        self.mpool = MaxPoolLayer(leaf_actions)

        # Layer5: Softmax (Applied in 'forward' method)
        if self.action_net:
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Layer 1 Linear Layer
        x1 = self.linear_decision(x)
        x2 = self.linear_action(x)
        x = torch.cat(
            (x1, x2),
            dim=1)  # decision_nodes are trainable but action_nodes are fixed

        # Layer 2 Activation Layer (Relu(+D) and Relu(-D))
        relu_x = self.reluP(x)
        relu_neg_x = self.reluM(-x)
        x = torch.cat((relu_x, relu_neg_x), dim=1)  #

        # Layer 3 Linear Layer (w/o) activation to add various conditions (node of each leaves)
        x = self.linear_leaf(x)

        # Layer 4 MaxPool Layer to get the max value for each leaf node
        x = self.mpool(x)

        # Layer 5 Softmax to get the probability distribution over the action
        if self.action_net:
            x = self.softmax(x)
        return x

    def save_dtnet(self, fn):
        checkpoint = dict()
        mdl_data = dict()
        W = torch.cat((self.linear_decision.weight.data.detach(),
                       self.linear_action.weight.data.detach()))
        B = torch.cat((self.linear_decision.bias.data.detach(),
                       self.linear_action.bias.data.detach()))
        mdl_data['W'] = W
        mdl_data['B'] = B
        mdl_data['L'] = self.linear_leaf.weight.data
        mdl_data['A'] = self.leaf_actions
        mdl_data['input_dim'] = self.input_dim
        mdl_data['output_dim'] = self.output_dim
        mdl_data['num_decision_nodes'] = self.num_decision_nodes
        mdl_data['num_action_nodes'] = self.num_action_nodes

        checkpoint['model_data'] = mdl_data
        torch.save(checkpoint, fn)

    def load_dtnet(self, fn):
        model_checkpoint = torch.load(fn, map_location=torch.device('cpu'))
        model_data = model_checkpoint['model_data']
        W = model_data['W'].detach().clone().data.cpu().numpy()
        B = model_data['B'].detach().clone().data.cpu().numpy()
        L = model_data['L'].detach().clone().data.cpu().numpy()
        A = model_data['A']
        num_decision_nodes = model_data['num_decision_nodes']
        num_action_nodes = model_data['num_action_nodes']
        input_dim = model_data['input_dim']
        output_dim = model_data['output_dim']
        dtnet_model = DTNetv1(input_dim=input_dim,
                              output_dim=output_dim,
                              node_weights=W,
                              node_bias=B,
                              leaf_weights=L,
                              leaf_actions=A,
                              init_leaves=self.init_leaves,
                              num_decision_nodes=num_decision_nodes,
                              num_action_nodes=num_action_nodes)
        return dtnet_model



#==== DTNet Older Version ====
class DTNetv0(nn.Module):
    """DTNet Architecture for the Decision Tree Network."""
    def __init__(self, input_dim: int, output_dim:int, node_weights: np.ndarray,
                 node_bias: np.ndarray, leaf_weights: np.ndarray,
                 leaf_actions: np.ndarray, init_leaves: t.List,
                 random: bool = False):
        super(DTNetv0, self).__init__()

        self.in_features = input_dim  # input dimension of state
        self.input_dim = input_dim  # input dimension of state # SPP: refactor this
        self.output_dim = output_dim  # output dimension of action
        self.out_features = len(
            node_bias)  # number of nodes in layer1 = number of nodes in DT
        self.num_leaves = len(leaf_actions)  # number of leaves in DT
        self.init_leaves = init_leaves  # Leaf Nodes as list: Used to getback the DT from DTNet
        self.random = random  # random initialization of weights and biases

        # Layer1: Linear Layer
        self.linear1 = nn.Linear(self.in_features, self.out_features)
        self.init_weights = torch.tensor(node_weights, dtype=torch.float32)
        self.init_biases = torch.tensor(node_bias, dtype=torch.float32)
        self.linear1.weight.data = self.init_weights  # intialize weights from DT weights
        self.linear1.bias.data = self.init_biases  # initialize weights from DT bias

        # Layer2: Relu Activation for Relu(+D) and Relu(-D)
        self.reluP = nn.ReLU()
        self.reluM = nn.ReLU()

        # Layer3: Linear Layer (w/o) activation to add various conditions
        self.linear2 = nn.Linear(2 * self.out_features,
                                 self.num_leaves,
                                 bias=False)
        self.init_weights2 = torch.tensor(leaf_weights, dtype=torch.float32)
        self.linear2.weight.data = self.init_weights2  # intialize weights from DT
        self.linear2.weight.requires_grad = False  # Fixed weights (no gradient)

        # RANDOM INITIALIZATION
        if self.random:

            # Random initialization of weights (Orthogonal Initialization)
            nn.init.orthogonal_(self.linear1.weight)
            # nn.init.xavier_uniform_(self.linear1.weight)
            # nn.init.uniform_(self.linear1.bias, a=-1, b=1)
            # # Random initialization of biases
            # if self.input_dim <= 8: # for carpole and lunar lander
            #     nn.init.uniform_(self.linear1.bias, a=-1, b=1)
            # else:
            #     nn.init.uniform_(self.linear1.bias, a=-40, b=40)

            # linear_action.weight is fixed (no gradient)

        # Layer4: MaxPool Layer
        self.leaf_actions = leaf_actions
        self.mpool = MaxPoolLayer(leaf_actions)

        # Layer5: Softmax (Applied in 'forward' method)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Layer 1 Linear Layer
        x = self.linear1(x)

        # Layer 2 Activation Layer (Relu(+D) and Relu(-D))
        relu_x = self.reluP(x)
        relu_neg_x = self.reluM(-x)
        x = torch.cat((relu_x, relu_neg_x), dim=1)  #

        # Layer 3 Linear Layer (w/o) activation to add various conditions (node of each leaves)
        x = self.linear2(x)

        # Layer 4 MaxPool Layer to get the max value for each leaf node
        x = self.mpool(x)

        # Layer 5 Softmax to get the probability distribution over the action
        x = self.softmax(x)
        return x

    def save_dtnet(self, fn):
        checkpoint = dict()
        mdl_data = dict()
        mdl_data['W'] = self.linear1.weight.data
        mdl_data['B'] = self.linear1.bias.data
        mdl_data['L'] = self.linear2.weight.data
        mdl_data['A'] = self.leaf_actions
        mdl_data['input_dim'] = self.input_dim
        mdl_data['output_dim'] = self.output_dim

        checkpoint['model_data'] = mdl_data
        torch.save(checkpoint, fn)

    def load_dtnet(self, fn):
        model_checkpoint = torch.load(fn) #, map_location=torch.device('cpu'))
        model_data = model_checkpoint['model_data']
        W = model_data['W'].detach().clone().data.cpu().numpy()
        B = model_data['B'].detach().clone().data.cpu().numpy()
        L = model_data['L'].detach().clone().data.cpu().numpy()
        A = model_data['A']
        input_dim = model_data['input_dim']
        output_dim = model_data['output_dim']
        dtnet_model = DTNetv0(input_dim=input_dim,
                            output_dim=output_dim,
                            node_weights=W,
                            node_bias=B,
                            leaf_weights=L,
                            leaf_actions=A,
                            init_leaves=self.init_leaves)
        return dtnet_model
