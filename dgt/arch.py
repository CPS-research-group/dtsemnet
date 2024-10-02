import itertools
import time
from collections import OrderedDict
from datetime import timedelta as td
from typing import (Any, Callable, Dict, Iterator, List, Optional, Tuple, Type,
                    Union, cast)

import numpy as np

import torch
import torch.nn as nn



from dgt.helper import ScaleBinarizer1, Sparser1, XLinear

"""
The Regression supported in this version of DGT is state independent (single theta parameter).
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
class DGT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        height: int,
        is_regression: bool = False,
    ):
        super().__init__()

        self._height = height # 63 internal nodes

        self._and_act_fn = torch.nn.Softmax(dim=-1)
        self._fp_type = torch.float32

        self._over_param = [] # [16, 16] # SPP: No overparametrization since DTSemNet also doesn't have it

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.is_regression = is_regression
        int_nodes = 2 ** height - 1 # SPP: define internal nodes
        leaf_nodes = 2 ** height

        

        ########### L1 ##########
        if len(self._over_param)==0:
            self._predicate_l = nn.Linear(in_dim, int_nodes)
            if True:
                with torch.no_grad(): nn.init.zeros_(self._predicate_l.bias)
            self._predicate_l = nn.Sequential(self._predicate_l)
        else:
            self._predicate_l = []
            mid_nodes = [int(x*int_nodes) for x in self._over_param]
            self._predicate_l = [nn.Linear(a,b) for a,b in zip([in_dim]+mid_nodes, mid_nodes+[int_nodes])]
            if True:
                with torch.no_grad():
                    [nn.init.zeros_(x.bias) for x in self._predicate_l if isinstance(x, nn.Linear)]
            self._predicate_l = nn.Sequential(*self._predicate_l)

        # orthogonal initialization of self._predicate_l all layer
        # For Fair Comparison
        for module in self._predicate_l.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)


        ########### L2 ##########
        weight, fixed_bias = DGT._get_and_layer_params(height, self._fp_type)
        # fixed_bias is now 0 not -h # SPP: ok
        # SPP: But I don't get why we need a bias term instead of 'h'.
        self._and_l = XLinear(int_nodes, leaf_nodes, weight=weight, bias=None, same=False)

        ########### L3 ##########
        self._or_l = XLinear(leaf_nodes, out_dim, bias=None, same=True)
        self.action_stds = nn.Parameter(torch.randn(out_dim, leaf_nodes), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.action_stds)


    def get_parameters_set(self, set_idx: int) -> Iterator[nn.Parameter]:
        if set_idx == 1:
            vals = [self._predicate_l.parameters()]

            return itertools.chain(*vals)

        elif set_idx == 2:
            return self._or_l.parameters()

        else:
            raise ValueError(f'{set_idx} must be in [1, 2]')

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor: # type: ignore


        ########### L1 ##########
        pred_z = self._predicate_l(x)

        pred_a, fac = ScaleBinarizer1.apply(pred_z) ; fac = fac.detach()
        pred_a = 2*pred_a - fac



        ########### L2 ##########
        and_z_a = self._and_l(pred_a)
        and_a = self._and_act_fn(and_z_a)
        and_a = Sparser1.apply(and_a)

        ########### L3 ##########
        or_z = self._or_l(and_a) # and expect CrossEntropyLoss (== log(softmax(x)))



        # SPP: For regression; Regression may not be useful if we assume controllers
        if self.is_regression:
            out = or_z.squeeze(1)
        else: # SPP: For classification
            # out softmax of or_z
            out = torch.nn.functional.softmax(or_z, dim=-1)

        std = torch.sum(and_a.unsqueeze(1) * self.action_stds, dim=-1)
        std = torch.clamp(std, min=-20, max=2)

        # print(std.shape, out.shape)
        # need to return mean and std    
        return out, std

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
