from agents.dtnet.agent import DTNetAgent
import gym
import numpy as np
import torch
from sklearn.metrics import r2_score


torch.set_printoptions(precision=2, sci_mode=False)

def test_dt_to_dtnet():
    agent = DTNetAgent(env_name='lunar',
                        path='dtnetlunar',
                        dt_type='minimal',
                        input_dim=8,
                        output_dim=4,
                        use_gpu=False,
                        epsilon=None,
                        epsilon_decay=None,
                        epsilon_min=None,
                        deterministic=True,
                        random=False)

    agent.action_network.eval()
    x = torch.randn(1000, 8)
    dt_act = agent.dt.get_action(x)
    with torch.no_grad():
        dtnet_act = agent.action_network.forward(x)
    dtnet_act = torch.argmax(dtnet_act, dim=1)

    # VALIDATE
    assert torch.all(dt_act == dtnet_act), 'DT to DTNet conversion failed'
    print('DT to DTNet conversion successful')
    # Tensors are equal
    # MSE Loss
    loss = torch.nn.MSELoss()
    l = loss(dt_act, dtnet_act)
    print("MSE Loss:", l.item())
    # compute R-squared
    r2 = r2_score(dt_act, dtnet_act)
    print("R-squared:", r2)