import numpy as np
import os
from monitor.eval_agent import test_sc_agent
from agents.dtnet.agent import DTNetAgent
from agents.fcnn.agent import FCNNAgent
from agents.prolonet.agent import DeepProLoNet
import gym
import datetime
import pytest

pytest.importorskip('test_sc_agent')

ENV_TYPE = 'zerlings'
dim_in = 37
dim_out = 10
PATH = 'dummy_path'
DT_TYPE = 'l2'
USE_GPU = False

policy_agent = DTNetAgent(  env_name=ENV_TYPE,
                                path=PATH,
                                dt_type=DT_TYPE,
                                use_gpu=USE_GPU,
                                input_dim=dim_in,
                                output_dim=dim_out,
                                random=False,
                                deterministic=True)


def test_sc2_install():
    reward = test_sc_agent(seed=10, policy_agent=policy_agent)
    
