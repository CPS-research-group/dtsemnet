import itertools
import time
from collections import OrderedDict
from datetime import timedelta as td
from typing import (Any, Callable, Dict, Iterator, List, Optional, Tuple, Type,
                    Union, cast)

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


class XLinear(nn.Module):
    """
    Provides more options to nn.Linear.

    If 'weight' is not None, fixes the weights of the layer to this.

    If 'bias' is None, it means that bias is learnable. In this case, whether all bias units
    should have the same bias or not is given by 'same'.

    If 'bias' is not None, then the provided value is assumed to the fixed bias (that is not
    updated/learnt). The value of 'same' is ignored here.

    Notes:
        - Number of neurons is out_features
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: Optional[torch.Tensor]=None,
        bias: Optional[torch.Tensor]=None,
        same: bool=False
    ):
        super().__init__()

        self._l = nn.Linear(in_features, out_features, bias=False)

        if weight is not None:
            self._l.weight = nn.Parameter(weight, requires_grad=False)

        if bias is None:
            self._bias = self.get_initialized_bias(in_features, 1 if same else out_features)
        else:
            self._bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, x):
        return self._l(x) + self._bias

    @property
    def weight(self):
        return self._l.weight

    @property
    def bias(self):
        return self._bias
    
    @staticmethod
    def get_initialized_bias(
        in_features: int, out_features: int, initialization_mean: Optional[torch.Tensor]=None
    ) -> nn.Parameter:

        if initialization_mean is None:
            initialization_mean = torch.zeros((out_features,)).float()
        assert initialization_mean.shape == (out_features,)

        k = 1 / math.sqrt(in_features)
        lb = initialization_mean - k
        ub = initialization_mean + k
        init_val = torch.distributions.uniform.Uniform(lb, ub).sample().float() # type: ignore

        return nn.Parameter(init_val, requires_grad=True) # type: ignore

class ScaleBinarizer1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        fac = torch.norm(input,p=1)/np.prod(input.shape)
        ctx.mark_non_differentiable(fac)
        return  (input >= 0).to(dtype=torch.float32)*fac, fac

    @staticmethod
    def backward(ctx, grad_output, grad_fac):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0
        return grad_input

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

class DGT(nn.Module):
    """
    Args:
    - learnable_and_bias: If True, bias in AND layer is learnt otherwise fixed to -counts[1]+eps
        and threshold*height is added later but before activation
    - learnable_or_bias: Bias in the OR layer is same for all neurons by default but if this is set to True,
        the bias is learnt otherwise it is set to 0 (for all neurons).

    Notes:
        In the AND layer for every neuron this is what happens:
        - When learnable_and_bias=False
            - ReLU(w.x - counts[1] + eps + threshold*height)
            - w is also fixed to {-1, 0, 1}
            - -counts[1] + eps is the fixed bias in XLinear, threshold*height is added in forward()
        - When learnable_and_bias=True
            - ReLU(w.x + b + height*threshold)
            - w is fixed, b is learnt
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        height: int,
        is_regression: False,
        over_param: list,
        linear_control: False,
        reg_hidden: int,
        _fp_type: torch.dtype=torch.float32,
    ):
        super().__init__()

        #fixed parameters
        learnable_and_bias = False
        learnable_or_bias = True
        self._batch_norm = False
        self._and_act_fn = nn.Softmax(dim=-1)
        
        self._height = height
        self._over_param = over_param
        self.is_regression = is_regression
        self._fp_type = _fp_type

        int_nodes = 2 ** height - 1
        leaf_nodes = 2 ** height

        self.linear_control = linear_control

        ########### L1 ##########


        if len(self._over_param)==0:
            self._predicate_l = nn.Linear(in_dim, int_nodes)
            with torch.no_grad(): nn.init.zeros_(self._predicate_l.bias)
            self._predicate_l = nn.Sequential(self._predicate_l)
        else:
            self._predicate_l = []
            mid_nodes = [int(x*int_nodes) for x in self._over_param]
            self._predicate_l = [nn.Linear(a,b) for a,b in zip([in_dim]+mid_nodes, mid_nodes+[int_nodes])]
            with torch.no_grad():
                [nn.init.zeros_(x.bias) for x in self._predicate_l if isinstance(x, nn.Linear)]
            self._predicate_l = nn.Sequential(*self._predicate_l)

        if self._batch_norm:
            self._predicate_bn = nn.BatchNorm1d(int_nodes)


        ########### L2 ##########
        weight, fixed_bias = DGT._get_and_layer_params(height, self._fp_type)
        # fixed_bias is now 0 not -h
        self._and_l = XLinear(int_nodes, leaf_nodes, weight=weight, bias=None if learnable_and_bias else fixed_bias, same=False)

        ########### L3 ##########
        self._or_l = XLinear(leaf_nodes, out_dim, bias=None if learnable_or_bias else torch.zeros((out_dim,)), same=True)

        if self.linear_control:
            if reg_hidden > 0:
                    hidden_layer = nn.Linear(in_dim, reg_hidden, bias=True)
                    output_layer = nn.Linear(reg_hidden, leaf_nodes, bias=True)
                    self.regression_layer = nn.Sequential(
                                                        hidden_layer,
                                                        output_layer
                                                    )
                
            else:
                self.regression_layer = nn.Linear(in_dim, leaf_nodes, bias=True)

    
    @staticmethod
    def _get_and_layer_params(height: int, fp_type) -> Tuple[torch.Tensor, torch.Tensor]:
        int_nodes = 2 ** height - 1
        leaf_nodes = 2 ** height

        weight = np.zeros((leaf_nodes, int_nodes))

        # Fill in the weight matrix level by level
        # i represents the level of nodes which we are handling at a given iteration
        for i in range(height):
            # Number of nodes in this level
            num_nodes = 2 ** i
            # Start number of node in this level
            start = 2 ** i - 1

            # Iterate through all nodes at this level
            for j in range(start, start + num_nodes):
                row_begin = (leaf_nodes // num_nodes) * (j - start)
                row_mid = row_begin + (leaf_nodes // (2 * num_nodes))
                row_end = row_begin + (leaf_nodes // num_nodes)

                weight[row_begin: row_mid, j] = 1
                weight[row_mid: row_end, j] = -1

        fixed_bias = torch.zeros(size=(2 ** height,), dtype=fp_type)

        return torch.from_numpy(weight).to(fp_type), fixed_bias

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor: 
        
        #### RESHAPE Image into vector ####
        x = x.view(x.size(0), -1)
        
        ########### L1 ##########
        pred_z = self._predicate_l(x)
        if self.linear_control:
            y = self.regression_layer(x)


        if self._batch_norm:
            pred_z = self._predicate_bn(pred_z)

        pred_a, fac = ScaleBinarizer1.apply(pred_z) ; fac = fac.detach()
        pred_a = 2*pred_a - fac

        ########### L2 ##########
        and_z_a = self._and_l(pred_a)
        and_a = self._and_act_fn(and_z_a)
        and_a_hard = Sparser1.apply(and_a)

        ########### L3 ##########
        or_z = self._or_l(and_a_hard) # and expect CrossEntropyLoss (== log(softmax(x)))
        
        if self.is_regression:
            if self.linear_control:
                or_z = and_a_hard - 0.5*and_a.detach() + 0.5*and_a # x_hard is the hard version of x
                or_z = or_z * y
                or_z = or_z.sum(dim=-1).reshape(-1, 1)
                # or_z = F.softmax(or_z, dim=-1)
            else:
                # for regression sqeez the 1
                
                or_z = or_z.squeeze(1)

        return or_z # no softmax in the output


def DGT8():
    return DGT(in_dim=784, out_dim=10, height=8, is_regression=False, over_param=[])