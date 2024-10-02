import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn



class Utils():
    @staticmethod
    def get_initialized_bias(
            in_features: int,
            out_features: int,
            initialization_mean: Optional[torch.Tensor] = None
    ) -> nn.Parameter:

        if initialization_mean is None:
            initialization_mean = torch.zeros((out_features, )).float()
        assert initialization_mean.shape == (out_features, )

        k = 1 / math.sqrt(in_features)
        lb = initialization_mean - k
        ub = initialization_mean + k
        init_val = torch.distributions.uniform.Uniform(
            lb, ub).sample().float()  # type: ignore

        return nn.Parameter(init_val, requires_grad=True)  # type: ignore


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
            self._bias = Utils.get_initialized_bias(in_features, 1 if same else out_features)
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
