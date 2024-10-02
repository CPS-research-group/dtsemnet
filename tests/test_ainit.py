import pytest



## Write test cased to test the conversion of dt to nn
ENV_TYPE = 'lunar'
PATH = 'dummy_path'
DT_TYPE = 'minimal'
dim_in = 8
dim_out = 4
USE_GPU = False
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
RANDOM = False


def test_init_dtnet():
    from agents.dtnet.agent import DTNetAgent
    policy_agent = DTNetAgent(env_name=ENV_TYPE,
                              path=PATH,
                              dt_type=DT_TYPE,
                              input_dim=dim_in,
                              output_dim=dim_out,
                              use_gpu=USE_GPU,
                              epsilon=EPSILON,
                              epsilon_decay=EPSILON_DECAY,
                              epsilon_min=EPSILON_MIN,
                              deterministic=False)
    print('DTNetAgent initialized successfully')

def test_init_prolo():
    from agents.prolonet.agent import DeepProLoNet
    policy_agent = DeepProLoNet(distribution='one_hot',
                                        path=PATH,
                                        input_dim=dim_in,
                                        output_dim=dim_out,
                                        use_gpu=USE_GPU,
                                        vectorized=False,
                                        randomized=RANDOM,
                                        adversarial=False,
                                        deepen=True,
                                        epsilon=EPSILON,
                                        epsilon_decay=EPSILON_DECAY,
                                        epsilon_min=EPSILON_MIN,
                                        deterministic=False)
    print('DeepProLoNet initialized successfully')

def test_init_fcnn():
    from agents.fcnn.agent import FCNNAgent
    policy_agent = FCNNAgent(env_name=ENV_TYPE,
                             path=PATH,
                             input_dim=dim_in,
                             output_dim=dim_out,
                             use_gpu=USE_GPU,
                             epsilon=EPSILON,
                             epsilon_decay=EPSILON_DECAY,
                             epsilon_min=EPSILON_MIN,
                             deterministic=False)
    print('FCNNAgent initialized successfully')



