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

# from entmax import sparsemax, entmax15, entmax_bisect


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
    def backward(ctx, grad_output, method='softmax'):
        
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


class DTSemNetReg(nn.Module):
    """DTNet Architecture for the Decision Tree Network.
    It outputs the weight of controller for each leaf node."""
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 node_weights: np.ndarray,
                 node_bias: np.ndarray,
                 leaf_weights: np.ndarray,
                 leaf_actions: np.ndarray,
                 init_leaves: t.List,
                 lin_control: bool = True,
                 num_controls: int = 1,
                 hk: bool = False
                 ):
        super(DTSemNetReg, self).__init__()
        self.lin_control = lin_control # whether to use linear control
        self.input_dim = input_dim  # input dimension of state # SPP: refactor this
        self.output_dim = output_dim  # number of leaves
        self.hk = hk # human knowledge
        self.out_features = len(
            node_bias)  # number of nodes in layer1 = number of nodes in DT
        self.num_leaves = len(leaf_actions)  # number of leaves in DT
        self.init_leaves = init_leaves  # Leaf Nodes as list: Used to getback the DT from DTNet
        
        self.num_controls = num_controls # number of controller (dim of action space of RL environment)

        # Layer1: Linear Layer
        self.linear1 = nn.Linear(self.input_dim, self.out_features)
        # todo: overparameterized can be added here (copy from dtnet_img)
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
        # Random initialization of weights (Orthogonal Initialization)
        nn.init.orthogonal_(self.linear1.weight, np.sqrt(2))
        nn.init.constant_(self.linear1.bias, 0.0)

        # Layer4: MaxPool Layer
        self.leaf_actions = leaf_actions
        self.mpool = MaxPoolLayer(leaf_actions)

        # Layer5: Softmax (Applied in 'forward' method)
        self.softmax = nn.Softmax(dim=-1)
        
        # layer5: Straight Through Estimator
        self.ste = STE()

        # layer6: controller (network output is the controller weight of each leaf)
        
        if self.lin_control:
            ## ====> Type 1
            # self.regression_layer_action = nn.ModuleList()
            # for i in range(num_controls):
            #     regression_layer = nn.Sequential(
            #         nn.Linear(output_dim, input_dim, bias=True)
            #     )
            #     for module in regression_layer.modules():
            #         if isinstance(module, nn.Linear):
            #             torch.nn.init.xavier_uniform_(module.weight)
            #     self.regression_layer_action.append(regression_layer)
            
            ## ====> Type 2
            self.regression_layer_action = nn.Linear(output_dim, self.num_controls*self.input_dim, bias=True)
            torch.nn.init.xavier_uniform_(self.regression_layer_action.weight)
        
        else:
            self.regression_layer_action = nn.ModuleList()
            self.num_controls = 1
            regression_layer = nn.Parameter(torch.randn(output_dim, 1), requires_grad=True)
            torch.nn.init.xavier_uniform_(regression_layer)
            self.regression_layer_action.append(regression_layer)
        

        # ===std
        self.action_stds = nn.Parameter(torch.randn(self.num_controls, self.output_dim), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.action_stds)

            


    def forward(self, in_x):
        
        # Layer 1 Linear Layer
        x = self.linear1(in_x)

        # Layer 2 Activation Layer (Relu(+D) and Relu(-D))
        relu_x = self.reluP(x)
        relu_neg_x = self.reluM(-x)
        x = torch.cat((relu_x, relu_neg_x), dim=1)  #

        # Layer 3 Linear Layer (w/o) activation to add various conditions (node of each leaves)
        x = self.linear2(x)

        # Layer 4
        # mpool layer not required

        # Layer 5 Softmax to get the probability distribution over the action
        x = self.softmax(x) # softmax output is converted to one-hot encoding in STE
        x_hard = self.ste(x)

        # straight through estimator
        x_op = x_hard - x.detach() + x
        # print(x)
        
        ### ===== > Type 1
        #---- Layer 6: Regression Layer
        # reg_op = torch.empty(x.shape[0], 0)
        # std = torch.empty(x.shape[0], 0)
        # for i, layer in enumerate(self.regression_layer_action):
        #     tmp_std = x_op * self.action_stds[i]
        #     tmp_std = tmp_std.sum(dim=-1).reshape(-1, 1)
        #     std = torch.cat((std, tmp_std), dim=1)  # Concatenate tmp_std to std
        
        #     reg = layer(x_op) * in_x
        #     reg = reg.sum(dim=-1).reshape(-1, 1)
        #     reg_op = torch.cat((reg_op, reg), dim=1)  # Concatenate reg to reg_op
        #-----
        
        ##======> Type 2
        ## Assuming self.regression_layer_action is a single layer with output shape (batch_size, 24 * 4)
        ##-----
        if self.lin_control:
            reshaped_layer_output = self.regression_layer_action(x_op).view(-1, self.num_controls, self.input_dim)
            reg_op = torch.sum(reshaped_layer_output * in_x.unsqueeze(1), dim=-1)
            
        else:
            # todo: check this
            reg_op = torch.sum(x_op.unsqueeze(1) * self.regression_layer_action.t(), dim=-1)
        
        std = torch.sum(x_op.unsqueeze(1) * self.action_stds, dim=-1)
        ##------
        
        std = torch.clamp(std, min=-20, max=2)        
        return reg_op, std

    def save_dtnet(self, fn):
        checkpoint = dict()
        mdl_data = dict()
        mdl_data['W'] = self.linear1.weight.data
        mdl_data['B'] = self.linear1.bias.data
        mdl_data['L'] = self.linear2.weight.data
        mdl_data['A'] = self.leaf_actions
        mdl_data['input_dim'] = self.input_dim
        mdl_data['output_dim'] = self.output_dim
        mdl_data['lin_control'] = self.lin_control
        mdl_data['hk'] = self.hk

        checkpoint['model_data'] = mdl_data
        torch.save(checkpoint, fn)

    def load_dtnet(self, fn):
        model_checkpoint = torch.load(fn, map_location=torch.device('cpu'))
        model_data = model_checkpoint['model_data']
        W = model_data['W'].detach().clone().data.cpu().numpy()
        B = model_data['B'].detach().clone().data.cpu().numpy()
        L = model_data['L'].detach().clone().data.cpu().numpy()
        A = model_data['A']
        input_dim = model_data['input_dim']
        output_dim = model_data['output_dim']
        try:
            lin_control = model_data['lin_control']
        except:
            lin_control = True
        dtnet_model = DTSemNetReg(input_dim=input_dim,
                              output_dim=output_dim,
                              node_weights=W,
                              node_bias=B,
                              leaf_weights=L,
                              leaf_actions=A,
                              init_leaves=self.init_leaves,
                              lin_control=lin_control)
        return dtnet_model

    def get_leaf(self, in_x):
        x = self.linear1(in_x)
        relu_x = self.reluP(x)
        relu_neg_x = self.reluM(-x)
        x = torch.cat((relu_x, relu_neg_x), dim=1)
        x = self.linear2(x)
        x = self.softmax(x)
        index = x.max(dim=-1, keepdim=True)[1]
        # x_hard = torch.zeros_like(x).scatter_(-1, index, 1.0)
        return index



class DTSemNetRegv0(nn.Module):
    """DTNet Architecture for the Decision Tree Network."""
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 node_weights: np.ndarray,
                 node_bias: np.ndarray,
                 leaf_weights: np.ndarray,
                 leaf_actions: np.ndarray,
                 init_leaves: t.List,
                 lin_control: bool = True,
                 num_controls: int = 1,
                 ):
        super(DTSemNetReg, self).__init__()
        self.lin_control = lin_control # whether to use linear control
        self.in_features = input_dim  # input dimension of state
        self.input_dim = input_dim  # input dimension of state # SPP: refactor this
        self.output_dim = output_dim  # number of leaves
        
        self.out_features = len(
            node_bias)  # number of nodes in layer1 = number of nodes in DT
        self.num_leaves = len(leaf_actions)  # number of leaves in DT
        self.init_leaves = init_leaves  # Leaf Nodes as list: Used to getback the DT from DTNet
        

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
        # Random initialization of weights (Orthogonal Initialization)
        nn.init.orthogonal_(self.linear1.weight, np.sqrt(2))
        nn.init.constant_(self.linear1.bias, 0.0)

        # Layer4: MaxPool Layer
        self.leaf_actions = leaf_actions
        self.mpool = MaxPoolLayer(leaf_actions)

        # Layer5: Softmax (Applied in 'forward' method)
        self.softmax = nn.Softmax(dim=-1)


        # ====> Regression Layer (New Network)
        # 1. cart (4 input dim) and lane_keeping (12 input dim) and ring_accel (44 input dim)
        if self.input_dim == 4 or self.input_dim == 12 or self.input_dim == 44:
            # == State dependent parameters
            self.regression_layer_action1 = nn.Sequential(
                nn.Linear(self.input_dim, self.output_dim, bias=True)
                # nn.Linear(64, self.output_dim)
                )
            for module in self.regression_layer_action1.modules():
                if isinstance(module, nn.Linear):
                    # nn.init.orthogonal_(module.weight, np.sqrt(2))
                    # nn.init.constant_(module.bias, 0.0)
                    torch.nn.init.xavier_uniform_(module.weight)
            
            # num of actions = 1
            self.action_stds = nn.Parameter(torch.randn(1, self.output_dim), requires_grad=True)
            torch.nn.init.xavier_uniform_(self.action_stds)
            
        # 2. Lunar Lander
        # Lunar Lander/highway/intersection/racetrack has 2 actions: main engine thrust and side engine thrust
        elif self.input_dim == 8 or self.input_dim == 25:
            # == State dependent parameters
            self.regression_layer_action1 = nn.Sequential(
                nn.Linear(self.input_dim, self.output_dim, bias=True)
                # nn.Linear(64, self.output_dim)
                )
            self.regression_layer_action2 = nn.Sequential(
                nn.Linear(self.input_dim,self.output_dim, bias=True),
                # nn.Linear(64, self.output_dim)
                )
            for module in self.regression_layer_action1.modules():
                if isinstance(module, nn.Linear):
                    # nn.init.orthogonal_(module.weight, np.sqrt(2))
                    # nn.init.constant_(module.bias, 0.0)
                    torch.nn.init.xavier_uniform_(module.weight)
            for module in self.regression_layer_action2.modules():
                if isinstance(module, nn.Linear):
                    # nn.init.orthogonal_(module.weight, np.sqrt(2))
                    # nn.init.constant_(module.bias, 0.0)
                    torch.nn.init.xavier_uniform_(module.weight)

            # == State Independent Parameters
            # self.static_actions = nn.Parameter(torch.randn(2, self.output_dim), requires_grad=True)
            # # nn.init.orthogonal_(self.static_actions, np.sqrt(2))
            # torch.nn.init.xavier_uniform_(self.static_actions)

            # num of actions = 2
            self.action_stds = nn.Parameter(torch.randn(2, self.output_dim), requires_grad=True)
            torch.nn.init.xavier_uniform_(self.action_stds)
        
        # 3. Walker, 24 input dim and 4 actions
        # Lunar Lander/highway/intersection/racetrack has 2 actions: main engine thrust and side engine thrust
        elif self.input_dim == 24:
            # == State dependent parameters
            if self.lin_control:
                self.regression_layer_action1 = nn.Sequential(
                    nn.Linear(self.input_dim, self.output_dim, bias=True)
                    # nn.Linear(64, self.output_dim)
                    )
                self.regression_layer_action2 = nn.Sequential(
                    nn.Linear(self.input_dim,self.output_dim, bias=True),
                    # nn.Linear(64, self.output_dim)
                    )
                self.regression_layer_action3 = nn.Sequential(
                    nn.Linear(self.input_dim,self.output_dim, bias=True),
                    # nn.Linear(64, self.output_dim)
                    )
                self.regression_layer_action4 = nn.Sequential(
                    nn.Linear(self.input_dim,self.output_dim, bias=True),
                    # nn.Linear(64, self.output_dim)
                    )
                for module in self.regression_layer_action1.modules():
                    if isinstance(module, nn.Linear):
                        # nn.init.orthogonal_(module.weight, np.sqrt(2))
                        # nn.init.constant_(module.bias, 0.0)
                        torch.nn.init.xavier_uniform_(module.weight)
                for module in self.regression_layer_action2.modules():
                    if isinstance(module, nn.Linear):
                        # nn.init.orthogonal_(module.weight, np.sqrt(2))
                        # nn.init.constant_(module.bias, 0.0)
                        torch.nn.init.xavier_uniform_(module.weight)
                for module in self.regression_layer_action3.modules():
                    if isinstance(module, nn.Linear):
                        # nn.init.orthogonal_(module.weight, np.sqrt(2))
                        # nn.init.constant_(module.bias, 0.0)
                        torch.nn.init.xavier_uniform_(module.weight)
                for module in self.regression_layer_action4.modules():
                    if isinstance(module, nn.Linear):
                        # nn.init.orthogonal_(module.weight, np.sqrt(2))
                        # nn.init.constant_(module.bias, 0.0)
                        torch.nn.init.xavier_uniform_(module.weight)
            else:
                # == State Independent Parameters
                self.static_actions = nn.Parameter(torch.randn(4, self.output_dim), requires_grad=True)
                # nn.init.orthogonal_(self.static_actions, np.sqrt(2))
                torch.nn.init.xavier_uniform_(self.static_actions)

            # num of actions = 4
            self.action_stds = nn.Parameter(torch.randn(4, self.output_dim), requires_grad=True)
            torch.nn.init.xavier_uniform_(self.action_stds)

            


    def forward(self, x):
        # regression layer
        if self.input_dim == 4 or self.input_dim == 12 or self.input_dim == 44:
            y = self.regression_layer_action1(x)

            std1 = self.action_stds
        elif self.input_dim == 8 or self.input_dim == 25:
            y1 = self.regression_layer_action1(x)
            y2 = self.regression_layer_action2(x)
            # print(y1, y2)
            #===== state independent parameters
            # y1 = self.static_actions[0]
            # y2 = self.static_actions[1]

            std1 = self.action_stds[0]
            std2 = self.action_stds[1]
        
        elif self.input_dim == 24: # walker
            if self.lin_control:
                y1 = self.regression_layer_action1(x)
                y2 = self.regression_layer_action2(x)
                y3 = self.regression_layer_action3(x)
                y4 = self.regression_layer_action4(x)
            else:
                #===== state independent parameters [a1_l1, a1_l2, ... a1_ln ], [a2_l1, a2_l2, ... a2_ln ], ...
                y1 = self.static_actions[0]
                y2 = self.static_actions[1]
                y3 = self.static_actions[2]
                y4 = self.static_actions[3]

            std1 = self.action_stds[0]
            std2 = self.action_stds[1]
            std3 = self.action_stds[2]
            std4 = self.action_stds[3]
        

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
        # info: Using novel entmax instead of softmax
        # x = sparsemax(x, dim=-1)
        # x = entmax15(x, dim=-1)
        # print(self.softmax(x))
        # print(sparsemax(x, dim=-1))
        # exit()

        # x: x_soft
        # Regression Layer
        # Straight through.
        index = x.max(dim=-1, keepdim=True)[1]

        x_hard = torch.zeros_like(x).scatter_(-1, index,
                                              1.0)  # one-hot encoding

        # straight through estimator
        x_op = x_hard - x.detach() + x

        
        # print(x_op)

        # For Pendulum and lane_keeping
        if self.input_dim == 4 or self.input_dim == 12 or self.input_dim == 44:
            reg_op = x_op * y
            reg_op = reg_op.sum(dim=-1).reshape(-1, 1)

            #=== find std
            output1_std = x_op * std1
            std = output1_std.sum(dim=-1).reshape(-1, 1)
            std = torch.clamp(std, min=-20, max=2)
        
        # For Lunar Lander
        elif self.input_dim == 8 or self.input_dim == 25:
            output1 = x_op * y1
            output1 = output1.sum(dim=-1).reshape(-1, 1)
            output2 = x_op * y2
            output2 = output2.sum(dim=-1).reshape(-1, 1)
            reg_op = torch.cat((output1, output2), dim=1)

            #=== find std
            output1_std = x_op * std1
            output1_std = output1_std.sum(dim=-1).reshape(-1, 1)
            output2_std = x_op * std2
            output2_std = output2_std.sum(dim=-1).reshape(-1, 1)
            std = torch.cat((output1_std, output2_std), dim=1)
            std = torch.clamp(std, min=-20, max=2)
        
        # For walker
        elif self.input_dim == 24:
            output1 = x_op * y1
            output1 = output1.sum(dim=-1).reshape(-1, 1)
            output2 = x_op * y2
            output2 = output2.sum(dim=-1).reshape(-1, 1)
            output3 = x_op * y3
            output3 = output3.sum(dim=-1).reshape(-1, 1)
            output4 = x_op * y4
            output4 = output4.sum(dim=-1).reshape(-1, 1)


            reg_op = torch.cat((output1, output2, output3, output4), dim=1)
            

            #=== find std
            output1_std = x_op * std1
            output1_std = output1_std.sum(dim=-1).reshape(-1, 1)
            output2_std = x_op * std2
            output2_std = output2_std.sum(dim=-1).reshape(-1, 1)
            output3_std = x_op * std3
            output3_std = output3_std.sum(dim=-1).reshape(-1, 1)
            output4_std = x_op * std4
            output4_std = output4_std.sum(dim=-1).reshape(-1, 1)
            
            std = torch.cat((output1_std, output2_std, output3_std, output4_std), dim=1)
            std = torch.clamp(std, min=-20, max=2)

            
        # print(self.static_actions)
        # print(self.action_stds)
        # print(index)
        
        return reg_op, std

    def save_dtnet(self, fn):
        checkpoint = dict()
        mdl_data = dict()
        mdl_data['W'] = self.linear1.weight.data
        mdl_data['B'] = self.linear1.bias.data
        mdl_data['L'] = self.linear2.weight.data
        mdl_data['A'] = self.leaf_actions
        mdl_data['input_dim'] = self.input_dim
        mdl_data['output_dim'] = self.output_dim
        mdl_data['lin_control'] = self.lin_control

        checkpoint['model_data'] = mdl_data
        torch.save(checkpoint, fn)

    def load_dtnet(self, fn):
        model_checkpoint = torch.load(fn, map_location=torch.device('cpu'))
        model_data = model_checkpoint['model_data']
        W = model_data['W'].detach().clone().data.cpu().numpy()
        B = model_data['B'].detach().clone().data.cpu().numpy()
        L = model_data['L'].detach().clone().data.cpu().numpy()
        A = model_data['A']
        input_dim = model_data['input_dim']
        output_dim = model_data['output_dim']
        try:
            lin_control = model_data['lin_control']
        except:
            lin_control = True
        dtnet_model = DTSemNetReg(input_dim=input_dim,
                              output_dim=output_dim,
                              node_weights=W,
                              node_bias=B,
                              leaf_weights=L,
                              leaf_actions=A,
                              init_leaves=self.init_leaves,
                              lin_control=lin_control)
        return dtnet_model


