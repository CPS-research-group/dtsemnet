import itertools
import time
from collections import OrderedDict
from datetime import timedelta as td
from typing import (Any, Callable, Dict, Iterator, List, Optional, Tuple, Type,
                    Union, cast)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F




import math
import numpy as np

def generate_complete_binary_tree(num_leaf, dim_out):
    leaf_action = [0] * dim_out #info: number of controllers is same as number of leaf nodes
    height = math.ceil(math.log2(num_leaf))
    leaf_nodes_lists = []
    stack = [(0, [], [])]

    controller_num = 0

    while stack:
        node, left_parents, right_parents = stack.pop()

        left_child = 2 * node + 1
        right_child = 2 * node + 2

        if len(left_parents) + len(right_parents) >= height:  # Leaf node
            leaf_act = leaf_action.copy()
            leaf_act[controller_num] = 1
            controller_num += 1
            if controller_num == dim_out:
                controller_num = 0
            leaf_nodes_lists.append([left_parents, right_parents, leaf_act])
        else:
            stack.append(
                (right_child, left_parents.copy(), right_parents + [node]))
            stack.append(
                (left_child, left_parents + [node], right_parents.copy()))

    assert len(leaf_nodes_lists) == num_leaf, 'The number of leaf nodes is not correct'
    return leaf_nodes_lists

def generate_complete_binary_tree_custom(num_leaf, dim_out, custom_leaf_actions):
    leaf_action = [0] * dim_out #info: number of controllers is same as number of leaf nodes
    height = math.ceil(math.log2(num_leaf))
    leaf_nodes_lists = []
    stack = [(0, [], [])]

    controller_num = 0

    while stack:
        node, left_parents, right_parents = stack.pop()

        left_child = 2 * node + 1
        right_child = 2 * node + 2

        if len(left_parents) + len(right_parents) >= height:  # Leaf node
            leaf_act = leaf_action.copy()
            leaf_act[custom_leaf_actions[controller_num]] = 1
            controller_num += 1
            leaf_nodes_lists.append([left_parents, right_parents, leaf_act])
        else:
            stack.append(
                (right_child, left_parents.copy(), right_parents + [node]))
            stack.append(
                (left_child, left_parents + [node], right_parents.copy()))

    assert len(leaf_nodes_lists) == num_leaf, 'The number of leaf nodes is not correct'
    
    return leaf_nodes_lists


def genDT(dim_in=784, dim_out=10, num_leaf=8, custom_leaf=None):
    """
    User defined DT for Lunar Lander environment. Returns node and leaf weights which represent a DT.

    Lunar lander with 7 internal nodes and 8 controllers

    Returns:
        dim_in: State dimension
        dim_out: Action dimension
        W: Node weights
        B: Node biases
        init_leaves: Leaf nodes
    """
    num_nodes = num_leaf - 1

    # 1. Define nodes of the tree of the form X.Wt + B > 0
    # State: X = [x0, x1, x2, x3, x4, x5, x6, x7]
    # W = np.random.randn(num_nodes, dim_in)  # each row represents a node
    # B = np.random.randn(num_nodes)  # Biases of each node

    W = np.random.randn(num_nodes, dim_in)  # each row represents a node
    B = np.random.randn(num_nodes)  # Biases of each node

    if custom_leaf:
        assert len(custom_leaf) == num_leaf, 'The number of leaf nodes is not correct'
        init_leaves = generate_complete_binary_tree_custom(num_leaf=num_leaf, dim_out=dim_out, custom_leaf_actions=custom_leaf)
        
    else:
        init_leaves = generate_complete_binary_tree(num_leaf=num_leaf, dim_out=dim_out)


    return dim_in, dim_out, W, B, init_leaves


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
        # out = torch.zeros((batch_size, self.num_groups)) #.to(x.device)
        # Initialize output tensor with zeros for each forward pass
        out = torch.zeros((batch_size, self.num_groups), dtype=x.dtype, device=x.device)

        # loop over node groups and compute the maximum value for each group
        for i, group in enumerate(self.node_groups):
            out[:, i], _ = torch.max(x[:, group], dim=1)
             

        return out



class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        max_values, max_indices = torch.max(input, dim=1, keepdim=True)  # Find the maximum value and its index across each sample in the batch
        output = torch.zeros_like(input)  # Initialize output tensor with zeros
        output.scatter_(1, max_indices, 1.0)  # Set 1 at the index of the maximum value for each sample
        # get softmax output
        op = F.softmax(input, dim=1)  # Apply softmax function
        ctx.save_for_backward(op, output)  # Save the output for backward pass
        return output

    
    @staticmethod
    def backward(ctx, grad_output, method='identity'):
        
        if method == 'softmax':
            output, output_hard = ctx.saved_tensors  # Retrieve the saved output from forward pass
            grad_input = grad_output * output * (1 - output)  # Propagate gradient through the softmax function
            # grad_input = torch.clamp(grad_input, min=-1, max=1)
        elif method == 'identity':
            grad_input = grad_output
            # grad_input = torch.clamp(grad_input, min=-1, max=1)
        elif method == 'hard':
            output, output_hard = ctx.saved_tensors  # Retrieve the saved output from forward pass
            grad_input = grad_output * output_hard
            # grad_input = torch.clamp(grad_input, min=-1, max=1)
        else:
            raise ValueError("Invalid backward propagation method")
        
        return grad_input

class STE(nn.Module):
    '''Straight Through Estimator'''
    def __init__(self):
        super(STE, self).__init__()

    def forward(self, x):
            x = STEFunction.apply(x)
            return x


class Sparser1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        inpargmax = input.argmax(-1)
        output = torch.zeros_like(input)
        output[torch.arange(input.shape[0]), inpargmax] = 1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

class NoiseSoftSparser1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, randbinprob):
        #ctx.save_for_backward(input)
        inpargmax = input.argmax(-1)
        mask = (torch.rand_like(input) < randbinprob).to(dtype=torch.float32)
        ctx.save_for_backward(input, mask)
        output = torch.zeros_like(input)
        output[torch.arange(input.shape[0]), inpargmax] = 1
        return (1-mask)*input + output*mask

    @staticmethod
    def backward(ctx, grad_output):
        input, mask = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input,None



class DTSemNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        is_regression: bool,
        height: int,
        over_param: list,
        linear_control: bool = False,
        wt_init: bool = False,
        reg_hidden: int = 0,
        custom_leaf = None,
    ):
        super().__init__()

        
        self._height = height
        self._over_param = over_param
        self.is_regression = is_regression
        self.linear_control = linear_control

        int_nodes = 2 ** height - 1
        leaf_nodes = 2 ** height

        if not is_regression: # classification
            assert out_dim > 1, "out_dim must be greater than 1 for classification"
            t_out_dim = out_dim
        else: # regression
            t_out_dim = leaf_nodes
            
        # Get DT and required parameters
        if custom_leaf:
            self.custom_leaf_unique = list(set(custom_leaf))
        else:
            self.custom_leaf_unique = None
        
        self.state_dim, self.action_dim, self.W, self.B, self.init_leaves = genDT(
                dim_in=in_dim, dim_out=t_out_dim,num_leaf=leaf_nodes, custom_leaf=custom_leaf)
        self.L, self.A = self.leaves_to_weight(self.init_leaves, self.W.shape[0])


        self.input_dim = in_dim  # input dimension of state 
        self.output_dim = out_dim  # output dimension of action
        self.out_features = len(self.B)  # number of nodes in layer1 = number of nodes in DT
        self.num_leaves = len(self.A)  # number of leaves in DT
        self.init_weights = torch.tensor(self.W, dtype=torch.float32)
        self.init_biases = torch.tensor(self.B, dtype=torch.float32)

        # Layer1: Linear Layer
        #info Overparams go here
        if len(self._over_param)==0:
            self.linear1 = nn.Linear(self.input_dim, self.out_features)
            # Random initialization of weights (Orthogonal Initialization)
            if wt_init:
                with torch.no_grad(): nn.init.zeros_(self.linear1.bias)
                self.linear1.weight.data = self.init_weights  # intialize weights from DT weights
                nn.init.orthogonal_(self.linear1.weight)
            self.linear1 = nn.Sequential(self.linear1) 
            
        else:
            self.linear1 = []
            mid_nodes = [int(x*int_nodes) for x in self._over_param]
            self.linear1 = [nn.Linear(a,b) for a,b in zip([in_dim]+mid_nodes, mid_nodes+[int_nodes])]
            with torch.no_grad():
                [nn.init.zeros_(x.bias) for x in self.linear1 if isinstance(x, nn.Linear)]
            self.linear1 = nn.Sequential(*self.linear1)
            
            # orthogonal initialization of linear1
            if wt_init:
                for m in self.linear1.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.orthogonal_(m.weight)
        

        # Layer2: Relu Activation for Relu(+D) and Relu(-D)
        self.reluP = nn.ReLU()
        self.reluM = nn.ReLU()

        # Layer3: Linear Layer (w/o) activation to add various conditions
        self.linear2 = nn.Linear(2 * self.out_features,
                                 self.num_leaves,
                                 bias=False)
        self.init_weights2 = torch.tensor(self.L, dtype=torch.float32)
        self.linear2.weight.data = self.init_weights2  # intialize weights from DT
        self.linear2.weight.requires_grad = False  # Fixed weights (no gradient)


        # Layer4: MaxPool Layer
        self.leaf_actions = self.A
        
        
        # regression layer:
        if self.is_regression:
            # Layer5: Softmax (Applied in 'forward' method)
            self.softmax = nn.Softmax(dim=-1)

            if self.linear_control:
                ### TYPE 1

                # when data set is 
                # 1. ctslice [64]
                # 2. year [32]
                # 3 abalone [16]
                # 4. pdb_bind None
                if reg_hidden > 0:
                    hidden_layer = nn.Linear(self.input_dim, reg_hidden, bias=True)
                    output_layer = nn.Linear(reg_hidden, t_out_dim, bias=True)
                    self.regression_layer = nn.Sequential(
                                                        hidden_layer,
                                                        output_layer
                                                    )
                    
                else:
                    self.regression_layer = nn.Linear(self.input_dim, t_out_dim, bias=True)
                    # torch.nn.init.uniform_(self.regression_layer.weight)

            else:
                self.regression_layer = nn.Linear(t_out_dim, 1, bias=True)
                # torch.nn.init.xavier_uniform_(self.regression_layer.weight)
        else:
            self.mpool = MaxPoolLayer(self.leaf_actions)
            
            

    def forward(
        self,
        in_x: torch.Tensor,
    ) -> torch.Tensor: # type: ignore

        in_x = in_x.view(in_x.size(0), -1)
        
        x = self.linear1(in_x)
        if self.is_regression:
            if self.linear_control:
                y = self.regression_layer(in_x)
                
            else:
                y = self.regression_layer
        

        # Layer 2 Activation Layer (Relu(+D) and Relu(-D))
        relu_x = self.reluP(x)
        relu_neg_x = self.reluM(-x)
        x = torch.cat((relu_x, relu_neg_x), dim=1)  #

        # Layer 3 Linear Layer (w/o) activation to add various conditions (node of each leaves)
        x = self.linear2(x)

        if self.is_regression: # for regression
            x = self.softmax(x)
            x_hard = Sparser1.apply(x)
            x = x_hard - 0.5*x.detach() + 0.5*x 
            
            
            if self.linear_control:
                x = x * y 
                x = x.sum(dim=-1).reshape(-1, 1)
                
            else:
                x = y(x)
            
            return x
        
        else: # for classification
            # Layer 4 MaxPool Layer to get the max value for each leaf node
            x = self.mpool(x)
            if self.custom_leaf_unique:
                batch_size, original_dim = x.size()
                expanded_dim = self.output_dim
                #
                expanded_output = torch.zeros(batch_size, expanded_dim, dtype=x.dtype, device=x.device)

                # Assign the original values to the specified indices
                expanded_output[:, self.custom_leaf_unique[:original_dim]] = x
                # print(expanded_output[1])
                # exit()
                expanded_output = torch.softmax(expanded_output, dim=-1)
                return expanded_output
            
            return x
    
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


