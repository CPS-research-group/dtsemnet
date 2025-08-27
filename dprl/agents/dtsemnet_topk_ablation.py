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
from entmax import entmax15



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

# Given DT Structure remove a particular leaf node and return the tree (retain the W and b)
def gen_DT_pruned(remove_leaf_ix, init_leaves, W, b):
    leaf = init_leaves[remove_leaf_ix]
    left_parents, right_parents, action = leaf
    l_max = max(left_parents) if len(left_parents) > 0 else 0
    r_max = max(right_parents) if len(right_parents) > 0 else 0

    if l_max > r_max:
        parent = left_parents[-1]
    else:
        parent = right_parents[-1]

    if parent == 0:
        print('Cannot prune root node')
        return W, b, init_leaves

    # remove this particular index from init_leaf and create a new_init_leaf
    new_init_leaves = []
    act = 0
    num_of_leaf = len(init_leaves) - 1 # one leaf removed
    for i, (left, right, action) in enumerate(init_leaves):
        # Skip the leaf being removed
        if i == remove_leaf_ix:
            continue

        # Remove the parent from left/right if it exists at the end
        if left and left[-1] == parent:
            left = left[:-1]
        if right and right[-1] == parent:
            right = right[:-1]

        # Decrement any value > parent in left and right
        left = [x - 1 if x > parent else x for x in left]
        right = [x - 1 if x > parent else x for x in right]

        # action = one-hot encoding of act
        action = list(np.eye(num_of_leaf, dtype=int)[act])
        act += 1
        new_init_leaves.append([left, right, action])
    
    # W_new = np.delete(W, parent, axis=0)
    # b_new = np.delete(b, parent, axis=0)
    W_new = torch.cat([W[:parent], W[parent+1:]], dim=0)
    b_new = torch.cat([b[:parent], b[parent+1:]], dim=0)

    
    return W_new, b_new, new_init_leaves

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

# regressor at each leaf
class Expert(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_dim=0):
        super(Expert, self).__init__()
        self.layer = nn.Linear(input_size, output_size)

        if hidden_dim > 0:
            self.layer = nn.Sequential(
                nn.Linear(input_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_size)
            )
            # Initialize weights and biases
        #     nn.init.kaiming_uniform_(self.layer[0].weight, nonlinearity='relu')
        #     nn.init.zeros_(self.layer[0].bias)
        #     nn.init.kaiming_uniform_(self.layer[2].weigh)
        #     nn.init.zeros_(self.layer[2].bias)
        # else:
        #     #Initialize weights and biases
        #     # nn.init.xavier_uniform_(self.layer.weight)
        #     nn.init.kaiming_uniform_(self.layer.weight)
        #     nn.init.zeros_(self.layer.bias)
    
    def forward(self, x):
        return self.layer(x)

class DTSemNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        height: int,
        over_param: list,
        ste: str = 'hard_argmax_ste',
        softmax_op: bool = False,
        is_regression: bool=True,
        linear_control: bool = False,
        wt_init: str = 'none',
        reg_hidden: int = 0,
        batch_norm: bool = True,
        custom_leaf = None,
        verbose: bool = False,
        top_k: int = 8,
        detach: bool = False,
        smax_temp: float = 1.0
    ):
        super().__init__()

        
        self._height = height
        self._over_param = over_param
        self.is_regression = is_regression
        self.linear_control = linear_control
        self.batch_norm = batch_norm
        self.softmax_op = softmax_op
        self.verbose = verbose
        self.reg_hidden = reg_hidden
        self.wt_init = wt_init
        self.ste = ste
        self.input_dim = in_dim  # input dimension of state 
        self.output_dim = out_dim  # output dimension of action
        self.temperature = smax_temp  # temperature for softmax
        

        self.top_k = top_k # number of experts to select, 1 for regression and >1 for MoE
        self.detach = detach

        self.internal_nodes = 2 ** height - 1
        self.num_leaves = 2 ** height

        if not is_regression: # classification
            assert self.output_dim > 1, "out_dim must be greater than 1 for classification"
            self.dtsemnet_out = self.output_dim
        else: # regression or all leaf as output
            self.dtsemnet_out = self.num_leaves
            # dtsemnet_out = 32
            
            
        # Get DT and required parameters
        if custom_leaf:
            self.custom_leaf_unique = list(set(custom_leaf))
        else:
            self.custom_leaf_unique = None
        
        self.state_dim, self.action_dim, self.W, self.B, self.init_leaves = genDT(
                dim_in=self.input_dim, dim_out=self.dtsemnet_out,num_leaf=self.num_leaves, custom_leaf=custom_leaf)
        self.L, self.A = self.leaves_to_weight(self.init_leaves, self.W.shape[0])
        self.init_weights = torch.tensor(self.W, dtype=torch.float32)
        self.init_biases = torch.tensor(self.B, dtype=torch.float32)

        # from dprl.utils.view_dt import build_tree, viz_tree
        # Tree = build_tree(self.init_leaves)
        # v = viz_tree(Tree[0], node_weights = self.W, node_biases= self.B)
        # v.write_png('dt_before_prune_regression.png')
        
        assert self.internal_nodes == len(self.B), "number of internal nodes should match len(B)"  # number of nodes in layer1 = number of nodes in DT
        assert self.num_leaves == len(self.A), 'Number of Leaf should match len(A)'  # number of leaves in DT
        

        self.init_arch(pruning=False) #initialize the architecture of the network
        

    def init_arch(self, pruning=False):
        # Layer1: Linear Layer
        #info Overparams go here
        if len(self._over_param)==0:
            self.linear1 = nn.Linear(self.input_dim, self.internal_nodes)
            # Random initialization of weights (Orthogonal Initialization)
            if not pruning:
                # if the wt_init is done then just copy the weights and biases
                # if self.wt_init=='none':
                #     with torch.no_grad(): nn.init.zeros_(self.linear1.bias)
                #     self.linear1.weight.data = self.init_weights  # intialize weights from DT weights
                    
                
                if self.wt_init == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(self.linear1.weight, nonlinearity='relu')
                    with torch.no_grad(): nn.init.zeros_(self.linear1.bias)
                    
                elif self.wt_init == 'kaiming_normal':
                    nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
                    with torch.no_grad(): nn.init.zeros_(self.linear1.bias)
                
                ## xavier works for tanh and sigmoid activations and ortho for deep and skip connections
                # elif self.wt_init == 'xavier_uniform':
                #     nn.init.xavier_uniform_(self.linear1.weight)
                # elif self.wt_init == 'xavier_normal':
                #     nn.init.xavier_normal_(self.linear1.weight)
                # elif self.wt_init == 'ortho':
                #     nn.init.orthogonal_(self.linear1.weight)
                
                # bias initialization
                
                     
            else:
                with torch.no_grad():
                    self.linear1.bias.copy_(self.init_biases)
                    self.linear1.weight.copy_(self.init_weights)

            self.linear1 = nn.Sequential(self.linear1)
            # print(self.linear1[0].bias)
            
        else:
            self.linear1 = []
            mid_nodes = [int(x*self.internal_nodes) for x in self._over_param]
            self.linear1 = [nn.Linear(a,b) for a,b in zip([self.input_dim]+mid_nodes, mid_nodes+[self.internal_nodes])]
            with torch.no_grad():
                [nn.init.zeros_(x.bias) for x in self.linear1 if isinstance(x, nn.Linear)]
            self.linear1 = nn.Sequential(*self.linear1)
            
            # orthogonal initialization of linear1
            if self.wt_init=='ortho':
                for m in self.linear1.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.orthogonal_(m.weight)
            elif self.wt_init=='xavier_norm':
                for m in self.linear1.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal_(m.weight)
            elif self.wt_init=='xavier_uniform':
                for m in self.linear1.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
        
        # if not pruning: # do initialize if batch norm is already initialized
        if self.batch_norm and not self.softmax_op:
            self.bn = nn.BatchNorm1d(self.internal_nodes)

        # Layer2: Relu Activation for Relu(+D) and Relu(-D)
        self.reluP = nn.ReLU()
        self.reluM = nn.ReLU()

        # Layer3: Linear Layer (w/o) activation to add various conditions
        self.linear2 = nn.Linear(2 * self.internal_nodes,
                                 self.num_leaves,
                                 bias=False)
        self.init_weights2 = torch.tensor(self.L, dtype=torch.float32)
        self.linear2.weight.data = self.init_weights2  # intialize weights from DT
        self.linear2.weight.requires_grad = False  # Fixed weights (no gradient)


        # Layer4: MaxPool Layer
        # self.leaf_actions = self.A
        self.softmax = nn.Softmax(dim=-1)
        self.mpool = MaxPoolLayer(self.A) # self.A is the action associated with each leaf
       
        if not pruning:
            # regression layer:
            if self.is_regression:
                # Layer5
                if self.linear_control:
                    self.experts = nn.ModuleList([Expert(self.input_dim, self.output_dim, self.reg_hidden) for _ in range(self.dtsemnet_out)])
                    self.selected_experts = None
                    self.log_stds = nn.Parameter(torch.randn(self.output_dim, self.num_leaves), requires_grad=True)
                    torch.nn.init.xavier_uniform_(self.log_stds)

            else:
                self.classification_layer = nn.Linear(self.dtsemnet_out, self.output_dim, bias=False)
                nn.init.xavier_uniform_(self.classification_layer.weight)
            
    def update_temperature(self, epoch, max_epoch=80, min_temp=0.6, max_temp=2.0):
        """Update temperature for softmax
        Linear annealing of temperature from max_temp to min_temp """
        self.temperature = max(min_temp, max_temp - (max_temp - min_temp) * (epoch / max_epoch))
        if self.verbose:
            print(f"Temperature: {self.temperature}")      

    
    
    def forward(
        self,
        in_x: torch.Tensor,
        mode='test'
    ) -> torch.Tensor: # type: ignore

        in_x = in_x.view(in_x.size(0), -1)
        # print(in_x)
        
        # Layer 1 Linear Layer which learns the node weights
        x = self.linear1(in_x)
        if self.batch_norm and not self.softmax_op:
            x = self.bn(x) # batch normalization of layer1 (internal nodes) for better training
        
        

        # Layer 2 Activation Layer (Relu(+D) and Relu(-D))
        relu_x = self.reluP(x)
        relu_neg_x = self.reluM(-x)
        x = torch.cat((relu_x, relu_neg_x), dim=1)  #

        # Layer 3 Linear Layer (w/o) activation to add various conditions (node of each leaves)
        x = self.linear2(x)
        # print('leaf', x)
        # Layer Regression Output
        if self.is_regression: # for regression
            # Layer 4: softmax across the leaf nodes
            batch_size = x.size(0)
            self.selected_experts = self.mpool(x)
            
            entropy_loss = None
            
            if self.linear_control:
                if mode == 'train':
                    # print('Training Mode')
                    self.top_expert = self.top_k
                    # gating_probs = self.softmax(x)
                    gating_probs = x
                    topk_vals, topk_inds = torch.topk(gating_probs, self.top_expert, dim=1)
                elif mode == 'aug':
                    # print('Augmentation Mode')
                    self.top_expert = 2  # only top-1 is finally used for output combination
                    gating_probs = x
                    topk_vals, topk_inds = torch.topk(gating_probs, self.top_expert, dim=1)
                elif mode=='test': # for test mode, select 1 expert (argmax)
                    # print('top1 Mode')
                    self.top_expert = 1 #select only one expert
                    # gating_probs = self.softmax(x)
                    gating_probs = x
                    topk_vals, topk_inds = torch.topk(gating_probs, self.top_expert, dim=1)
                else:
                    raise ValueError("Invalid mode. Use 'train', 'test' or 'aug'.")
                # print('topk_vals', topk_vals)
                # print('topk_inds', topk_inds)
                ### Type 3
                # Reshape input to [batch_size, 1, input_dim] -> broadcast to [batch_size, top_k, input_dim]
                in_x_expanded = in_x.unsqueeze(1).expand(-1, self.top_expert, -1)
                # print('in_x_expanded', in_x_expanded)

                # Flatten for processing: [batch_size * top_k, input_dim]
                flat_inputs = in_x_expanded.reshape(-1, in_x.shape[-1])
                flat_expert_inds = topk_inds.reshape(-1)
                # print('flat_inputs', flat_expert_inds)

                

                # after you compute topk_vals, topk_inds, in_x_expanded, flat_inputs, flat_expert_inds ...
                # Build a boolean "this slot is top-1" mask shaped like topk_inds, then flatten it
                is_top1_slot = torch.zeros_like(topk_inds, dtype=torch.bool)
                is_top1_slot[:, 0] = True                            # column 0 is the top-1 per sample (torch.topk is sorted desc)
                flat_is_top1 = is_top1_slot.reshape(-1)              # [batch_size * top_k]

                outputs   = torch.zeros(flat_inputs.size(0), self.output_dim, device=in_x.device)
                log_stds  = torch.zeros_like(outputs, device=in_x.device)

                for expert_id in range(len(self.experts)):
                    # Which flattened rows (sample, slot) want this expert?
                    mask = (flat_expert_inds == expert_id)          # [batch_size * top_k]
                    if not mask.any():
                        continue

                    selected_inputs = flat_inputs[mask]             # [n_rows_for_this_expert, in_dim]
                    y = self.experts[expert_id](selected_inputs)    # [n_rows_for_this_expert, out_dim]
                    # For rows that are NOT top-1 in their sample, stop gradient to this expert:
                    local_top1 = flat_is_top1[mask]                 # [n_rows_for_this_expert] boolean
                    # Mix attached vs detached outputs row-wise
                    y = torch.where(local_top1.unsqueeze(-1), y, y.detach())
                    outputs[mask] = y

                    # Same treatment for the expert-specific log_std parameter rows
                    ls = self.log_stds[:, expert_id].unsqueeze(0).expand(selected_inputs.size(0), -1)  # [n_rows, out_dim]
                    ls = torch.where(local_top1.unsqueeze(-1), ls, ls.detach())
                    log_stds[mask] = ls

                # (Optional) if you ALSO want to stop gradients to gating weights for non–top-1 slots:
                # topk_vals = torch.where(is_top1_slot, topk_vals, topk_vals.detach())

                if mode == 'aug':
                    # return the outputs and log_stds for all experts where batch size becomes double because of augmentation
                    # each sample has two outputs, one for each expert
                    return outputs, log_stds
                
                else:
                
                    # Reshape back to [batch_size, top_k, output_dim]
                    expert_outputs = outputs.view(batch_size, self.top_expert, self.output_dim)
                    expert_log_stds = log_stds.view(batch_size, self.top_expert, self.output_dim)
                    # print('self.log_stds', self.log_stds)
                    # print('expert_outputs', expert_outputs)
                    # print('expert_log_stds', expert_log_stds)
                    if self.detach: # this is also a kind of small assumptions but it is never used since it is not givig good results
                        # topk_vals = topk_vals / (topk_vals.sum(dim=1, keepdim=True).detach())
                        raise NotImplementedError("detach is not implemented yet")
                    else:
                        if self.temperature > 0:
                            topk_vals = F.softmax(topk_vals / self.temperature, dim=1) # sharp less entropy
                        else:
                            topk_vals = topk_vals / (topk_vals.sum(dim=1, keepdim=True)) # more entropy

                    #### INFO
                    # detach: creates dense leaves, dense leaves seems to be better
                    # 1. ailerons: detach
                    ####
                    # print('topk_vals', topk_vals)
                    topk_vals_expanded = topk_vals.unsqueeze(-1)
                    
                    weighted_sum = expert_outputs * (topk_vals_expanded)
                    combined_output = weighted_sum.sum(dim=1)

                    combined_log_std = (expert_log_stds * topk_vals_expanded).sum(dim=1)
                    combined_log_std = torch.clamp(combined_log_std, min=-20.0, max=2.0)


                    return combined_output, combined_log_std
                
            else: # linear control is False
                raise NotImplementedError("Implement scalar parameters at leaf nodes")
            
        # Layer Classification outpyt
        else: # for classification
            # Layer 4 MaxPool Layer to get the max value for each leaf node
            x = self.mpool(x) 
            # Layer 5 STE
            x = self.ste(x)
            # Layer 6 Classification Layer
            x = self.classification_layer(x)
            # Layer: Softmax for RL only
            if self.softmax_op:
                x = self.softmax(x) # in case of RL is expects probabilites
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


    def prune(self, data_loader, threshold: float = 0.01, device='cpu'):
        """
        Prunes leaves with average probability < threshold based on softmax routing scores.
        Args:
            data_loader: DataLoader object to iterate over data.
            threshold: Minimum average routing probability for a leaf to be retained.
            device: Device for model/data
        """
        print("---Pruning Leaves---")
        self.eval()
        total_counts = torch.zeros(self.num_leaves, device=device)

        with torch.no_grad():
            for batch,_,_ in data_loader:
                batch = batch.to(device)
                if isinstance(batch, (tuple, list)):
                    x = batch[0]
                else:
                    x = batch

                x = x.to(device)
                x = x.view(x.size(0), -1)

                # Forward until leaf probability layer
                x1 = self.linear1(x)
                if self.batch_norm and not self.softmax_op:
                    x1 = self.bn(x1)

                relu_x = self.reluP(x1)
                relu_neg_x = self.reluM(-x1)
                x_cat = torch.cat((relu_x, relu_neg_x), dim=1)
                leaf_logits = self.linear2(x_cat)

                probs = self.softmax(leaf_logits)  # shape: (batch_size, num_leaves)
                total_counts += probs.sum(dim=0)  # sum over batch

        avg_counts = total_counts / len(data_loader.dataset)

        # Find index of the leaf to prune (lowest probability)
        min_prob, prune_idx = torch.min(avg_counts, dim=0)

        # Only prune if the minimum probability is below the threshold
        if min_prob.item() < threshold:
            print(f"Pruning leaf {prune_idx} with avg prob {min_prob.item():.4f} < threshold {threshold}")
            # Proceed with pruning logic using `prune_idx`
        else:
            print(f"No leaf pruned. Minimum avg prob {min_prob.item():.4f} >= threshold {threshold}")
            return False  # or skip pruning
                # if self.verbose:
       
        print(f"Pruning Leaf {prune_idx}")

        # Compute a new tree structure after pruning
        W = self.linear1[0].weight.detach().cpu()
        B = self.linear1[0].bias.detach().cpu()
        print(B)    
        self.W, self.B, self.init_leaves = gen_DT_pruned(prune_idx, self.init_leaves, W, B)
        self.L, self.A = self.leaves_to_weight(self.init_leaves, self.W.shape[0])
        

        # update various tree parameters
        self.internal_nodes -= 1
        self.num_leaves -= 1
        self.dtsemnet_out = self.num_leaves
        
        assert self.internal_nodes == len(self.B), "number of internal nodes should match len(B)"  # number of nodes in layer1 = number of nodes in DT
        assert self.num_leaves == len(self.A), 'Number of Leaf should match len(A)'  # number of leaves in DT
        # self.init_weights = torch.tensor(self.W )
        # self.init_biases = torch.tensor(self.B)
        self.init_weights = self.W
        self.init_biases = self.B

        self.init_arch(pruning=True) #pruned architecture
        self.experts = nn.ModuleList([
                                        expert for i, expert in enumerate(self.experts) if i != prune_idx
                                    ])


        from dprl.utils.view_dt import build_tree, viz_tree
        Tree = build_tree(self.init_leaves)
        v = viz_tree(Tree[0], node_weights=self.W.detach().cpu().numpy(), node_biases=self.B.detach().cpu().numpy())
        v.write_png('dt_after_prune_regression.png')
        return True

    