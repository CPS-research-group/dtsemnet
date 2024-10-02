"""Architecutre for Fully Connected Neural Network.
It is used as a value network for the DTNetAgent.
"""

import torch.nn as nn
import torch.nn.functional as F
import logging


class FCNN(nn.Module):
    """Fully Connected Neural Network."""
    def __init__(self, input_dim=None, output_dim=None, env=None, action_net=False, arch=None, value_net_single_op=False):
        super(FCNN, self).__init__()

        self.env = env
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.action_net = action_net
        self.arch = arch
        self.value_net_single_op = value_net_single_op # single output for value net single or multiple outputs (DTNetAgent)


        if self.env =='cart':
            if self.arch == '32x32':
                logging.info('Using cartpole architecture 32x32')
                self.layer1 = nn.Linear(input_dim, 32)
                self.layer2 = nn.Linear(32, 32)
                self.activation = nn.ReLU()
                self.layer3 = nn.Linear(32, output_dim)
                # initialize layer weights
                nn.init.orthogonal_(self.layer1.weight)
                nn.init.orthogonal_(self.layer2.weight)
                nn.init.orthogonal_(self.layer3.weight)
            elif self.arch == '16x16':
                logging.info('Using cartpole architecture 16x16')
                self.layer1 = nn.Linear(input_dim, 16)
                self.layer2 = nn.Linear(16, 16)
                self.activation = nn.ReLU()
                self.layer3 = nn.Linear(16, output_dim)
                # initialize layer weights
                nn.init.orthogonal_(self.layer1.weight)
                nn.init.orthogonal_(self.layer2.weight)
                nn.init.orthogonal_(self.layer3.weight)
            elif self.arch == '16l':
                logging.info('Using cartpole architecture 16 linear')
                self.layer1 = nn.Linear(input_dim, 16)
                self.layer2 = nn.Linear(16, output_dim)
                # initialize layer weights
                nn.init.orthogonal_(self.layer1.weight)
                nn.init.orthogonal_(self.layer2.weight)
            elif self.arch == '16':
                logging.info('Using cartpole architecture 16 linear with ReLU')
                self.layer1 = nn.Linear(input_dim, 16)
                self.activation = nn.ReLU()
                self.layer2 = nn.Linear(16, output_dim)
                # initialize layer weights
                nn.init.orthogonal_(self.layer1.weight)
                nn.init.orthogonal_(self.layer2.weight)



        elif self.env == 'zerlings' :
            # ================= ARCHITECTURE [256, 256, 126] (Works for Zerglings) =================
            self.layer1 = nn.Linear(input_dim, 256)
            self.layer2 = nn.Linear(256, 256)
            self.layer3 = nn.Linear(256, 126)
            self.activation = nn.ReLU()
            self.layer4 = nn.Linear(126, output_dim)
            # initialize layer weights
            nn.init.orthogonal_(self.layer1.weight)
            nn.init.orthogonal_(self.layer2.weight)
            nn.init.orthogonal_(self.layer3.weight)
            nn.init.orthogonal_(self.layer4.weight)


        elif self.env in ['lunar', 'acrobot']:
            # # ================= ARCHITECTURE 64x64 (Works for Lunar Lander) =================
            if self.arch == '64x64':
                logging.info('Using 64x64 architecture')
                self.layer1 = nn.Linear(input_dim, 64)
                self.layer2 = nn.Linear(64, 64)
                self.activation = nn.ReLU()
                self.layer3 = nn.Linear(64, output_dim)
                # initialize layer weights
                nn.init.orthogonal_(self.layer1.weight)
                nn.init.orthogonal_(self.layer2.weight)
                nn.init.orthogonal_(self.layer3.weight)

            elif self.arch == '32x32':
                logging.info('Using 32x32 architecture')
                self.layer1 = nn.Linear(input_dim, 32)
                self.layer2 = nn.Linear(32, 32)
                self.activation = nn.ReLU()
                self.layer3 = nn.Linear(32, output_dim)
                # initialize layer weights
                nn.init.orthogonal_(self.layer1.weight)
                nn.init.orthogonal_(self.layer2.weight)
                nn.init.orthogonal_(self.layer3.weight)
            
            elif self.arch == '16x16':
                logging.info('Using 16x16 architecture')
                self.layer1 = nn.Linear(input_dim, 16)
                self.layer2 = nn.Linear(16, 16)
                self.activation = nn.ReLU()
                self.layer3 = nn.Linear(16, output_dim)
                # initialize layer weights
                nn.init.orthogonal_(self.layer1.weight)
                nn.init.orthogonal_(self.layer2.weight)
                nn.init.orthogonal_(self.layer3.weight)

            elif self.arch == '128x128':
                logging.info('Using 128x128 architecture')
                self.layer1 = nn.Linear(input_dim, 128)
                self.layer2 = nn.Linear(128, 128)
                self.activation = nn.ReLU()
                self.layer3 = nn.Linear(128, output_dim)
                # initialize layer weights
                nn.init.orthogonal_(self.layer1.weight)
                nn.init.orthogonal_(self.layer2.weight)
                nn.init.orthogonal_(self.layer3.weight)

            elif self.arch == '32':
                logging.info('Using 32 architecture with ReLU')
                self.layer1 = nn.Linear(input_dim, 32)
                self.activation = nn.ReLU()
                self.layer2 = nn.Linear(32, output_dim)
                # initialize layer weights
                nn.init.orthogonal_(self.layer1.weight)
                nn.init.orthogonal_(self.layer2.weight)
            
            elif self.arch == '32l':
                logging.info('Using 32node linear layer')
                self.layer1 = nn.Linear(input_dim, 32)
                self.layer2 = nn.Linear(32, output_dim)
                # initialize layer weights
                nn.init.orthogonal_(self.layer1.weight)
                nn.init.orthogonal_(self.layer2.weight)
            
            elif self.arch == '16l':
                logging.info('Using 16node linear layer')
                self.layer1 = nn.Linear(input_dim, 16)
                self.layer2 = nn.Linear(16, output_dim)
                # initialize layer weights
                nn.init.orthogonal_(self.layer1.weight)
                nn.init.orthogonal_(self.layer2.weight)
            
            elif self.arch == '16':
                logging.info('Using 16node linear layer with relu')
                self.layer1 = nn.Linear(input_dim, 16)
                self.activation = nn.ReLU()
                self.layer2 = nn.Linear(16, output_dim)
                # initialize layer weights
                nn.init.orthogonal_(self.layer1.weight)
                nn.init.orthogonal_(self.layer2.weight)

            elif self.arch == '64':
                logging.info('Using 64 architecture with ReLU')
                self.layer1 = nn.Linear(input_dim, 64)
                self.activation = nn.ReLU()
                self.layer2 = nn.Linear(64, output_dim)
                # initialize layer weights
                nn.init.orthogonal_(self.layer1.weight)
                nn.init.orthogonal_(self.layer2.weight)


            else:
                # Architecutre for Value Functions
                logging.info('Using 64x64x32 architecture')
                self.layer1 = nn.Linear(input_dim, 64)
                self.layer2 = nn.Linear(64, 64)
                self.layer3 = nn.Linear(64, 32)
                self.activation = nn.ReLU()
                self.layer4 = nn.Linear(32, output_dim)
                # orthogonal initialization
                nn.init.orthogonal_(self.layer1.weight)
                nn.init.orthogonal_(self.layer2.weight)
                nn.init.orthogonal_(self.layer3.weight)
                nn.init.orthogonal_(self.layer4.weight)


        else:
            raise NotImplementedError

        if action_net: # use softmax for action network
            self.softmax = nn.Softmax(dim=1)
        else: # use linear (1) output for value network
            if self.value_net_single_op:
                self.value_output = nn.Linear(output_dim, 1) # output single value



    def forward(self, x):
        if self.env == 'cart': # for 2 hidden layer architectures
            if self.arch in ['32x32', '16x16']:
                x = self.layer1(x)
                x = self.activation(x)
                x = self.layer2(x)
                x = self.activation(x)
                x = self.layer3(x)
            elif self.arch == '16l':
                x = self.layer1(x)
                x = self.layer2(x)
            elif self.arch == '16':
                x = self.layer1(x)
                x = self.activation(x)
                x = self.layer2(x)

        elif self.env in ['lunar', 'zerlings', 'acrobot']: # for 3 hidden layered architectures
            if self.arch == '64x64' or self.arch == '32x32' or self.arch == '128x128' or self.arch == '16x16':
                x = self.layer1(x)
                x = self.activation(x)
                x = self.layer2(x)
                x = self.activation(x)
                x = self.layer3(x)
               

            elif self.arch == '32' or self.arch == '64' or self.arch == '16':
                x = self.layer1(x)
                x = self.activation(x)
                x = self.layer2(x)

            elif self.arch in ['32l', '16l']:
                x = self.layer1(x)
                x = self.layer2(x)  

            else:
                x = self.layer1(x)
                x = self.activation(x)
                x = self.layer2(x)
                x = self.activation(x)
                x = self.layer3(x) # no activation for last layer
                x = self.activation(x)
                x = self.layer4(x)

        # action or value network output
        if self.action_net: # use sigmoid for action network
            x = self.softmax(x)
        else: # single value for value net
            if self.value_net_single_op:
                x = self.value_output(x)


        return x