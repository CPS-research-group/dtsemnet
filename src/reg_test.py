import pdb
import math
import argparse
import copy
import sys
from datetime import datetime
import json
import pickle


import time
import numpy as np


from rich import print


import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
torch.backends.cudnn.deterministic = True
## DTSemNet
from src.dtsemnet import DTSemNet
from src.dgt import DGT




model = DTSemNet(
                in_dim=4,
                out_dim=1,
                height=3,
                is_regression=True,
                over_param=[],
                linear_control=True,
                wt_init=False
                )

inp = torch.randn(2, 4)
tar = torch.randn(2, 1)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
out = model(inp)
loss = loss_fn(out, tar)
optimizer.zero_grad()
loss.backward()
print("Initial weights:")
print(model.regression_layer.weight)
print("Initial gradients:")
print(model.regression_layer.weight.grad)
optimizer.step()
print("Updated weights:")
print(model.regression_layer.weight)
print("Updated gradients:")
print(model.regression_layer.weight.grad)

