import numpy as np



config_abalone = {
    "code": "abalone",
    "name": "Abalone",
    "filepath": "datasets/abalone.data",
    "n_attributes": 10,
    "attributes": ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings'],
    "n_classes": 1,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 5e-4, #5e-4 #1e-3
    "grad_clip": True,
    "over_param": [8], #[8]
    "reg_hidden": 0,
    "nseed": 5, # 5
    "tseed": 1,
    "test_split": 0.4,
    "val_split": 0.2,
    "depth": 5,
    "ste": ["hard_argmax_ste", "gumbel_softmax_ste", "gumbel_max_ste", "relu_ste", "clipped_relu_ste", "entmax_ste"][-1],
    "bins": np.array([-2.82, -0.2951,]),
    "wt_init": 'none', # xavier normal weight init ['ortho, 'xavier_uniform', 'xavier_norm', 'none']
    "smax_temp": 0.5, # temperature for softmax, not used normalization
    
    "epochs": 20,
    "batch_size": 64,
    "learning_rate": 0.01,
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.90,

    "top_k": 4,
    "epochs_fine_tune": 10,
    "mix_experts": 2,
    "learning_rate_tune": 0.0001,
    "detach": False,
    "use_pretrained": False,
}

config_ailerons = {
    "code": "ailerons",
    "name": "Ailerons",
    "filepath": "datasets/ailerons/",
    "n_attributes": 40,
    "attributes": ['climbRate', 'Sgz', 'p', 'q', 'curPitch', 'curRoll', 'absRoll', 'diffClb', 'diffRollRate', 'diffDiffClb', 'SeTime1', 'SeTime2', 'SeTime3', 'SeTime4', 'SeTime5', 'SeTime6', 'SeTime7', 'SeTime8', 'SeTime9', 'SeTime10', 'SeTime11', 'SeTime12', 'SeTime13', 'SeTime14', 'diffSeTime1', 'diffSeTime2', 'diffSeTime3', 'diffSeTime4', 'diffSeTime5', 'diffSeTime6', 'diffSeTime7', 'diffSeTime8', 'diffSeTime9', 'diffSeTime10', 'diffSeTime11', 'diffSeTime12', 'diffSeTime13', 'diffSeTime14', 'alpha', 'Se', 'goal'],
    "n_classes": 1,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 5e-4, # 5e-4
    "over_param": [],
    "reg_hidden": 0,
    "nseed": 1,
    "tseed": 1,
    "grad_clip": True,
    "test_split": None,
    "val_split": 0.2,
    "depth": 5,
    "ste": ["hard_argmax_ste", "gumbel_softmax_ste", "gumbel_max_ste", "relu_ste", "clipped_relu_ste", "entmax_ste"][-1], # not used
    "bins": np.array([-7, -0.5695, 0.1649, 0.655]),
    "wt_init": 'none', # xavier normal weight init ['ortho, 'xavier_uniform', 'xavier_norm', 'none']
    "smax_temp": 0.5, # temperature for softmax, not used normalization
    
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 0.01, #0.01 for top1
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.90,

    "top_k": 4,
    "epochs_fine_tune": 15,
    "mix_experts": 2,
    "learning_rate_tune": 0.00001,
    "detach": False,
    "use_pretrained": False,

    # "epochs": 80,
    # "batch_size": 64,
    # "learning_rate": 0.04,
    # "lr_scheduler": True,
    # "lr_scheduler_gamma": 0.88
}

config_pdb_bind = {
    "code": "pdb_bind",
    "name": "PDB Bind",
    "filepath": "datasets/pdb_bind/",
    "n_attributes": 2052,
    "attributes": [],
    "n_classes": 1,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 0.01, 
    "nseed": 1,
    "over_param": [],
    "reg_hidden": 0,
    "tseed": None,
    "grad_clip": False,
    "test_split": None,
    "val_split": None,
    "depth": 2,
    "ste": ["hard_argmax_ste", "gumbel_softmax_ste", "gumbel_max_ste", "relu_ste", "clipped_relu_ste", "entmax_ste"][-1],
    "bins": np.array([-3.1, -0.0]),
    "wt_init": 'none',
    "smax_temp": 0, # temperature for softmax, not used in normalization
    
    "epochs": 60,
    "batch_size": 256,
    "learning_rate": 0.01,
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.95,

    "top_k": 2,
    "epochs_fine_tune": 0,
    "mix_experts": 2,
    "learning_rate_tune": 0,
    "detach": False,
    "use_pretrained": False,
}
# top 1.33


config_cpu_active = {
    "code": "cpu_active",
    "name": "CPU Active",
    "filepath": "datasets/cpu_active/",
    "n_attributes": 21,
    "attributes": ['time', 'lread', 'lwrite', 'scall', 'sread', 'swrite', 'fork', 'exec', 'rchar', 'wchar', 'pgout', 'ppgout', 'pgfree', 'pgscan', 'atch', 'pgin', 'ppgin', 'pflt', 'vflt', 'runqsz', 'runocc', 'freemem', 'freeswap', 'usr', 'sys', 'wio', 'idle'],
    "n_classes": 1,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 1e-4,
    "over_param": [],
    "reg_hidden": 0,
    "nseed": 5, # 5
    "tseed": 1,
    "grad_clip": True,
    "test_split": 0.4,
    "val_split": 0.2,
    "depth": 5,
    "ste": ["hard_argmax_ste", "gumbel_softmax_ste", "gumbel_max_ste", "relu_ste", "clipped_relu_ste", "entmax_ste"][-1],
    "bins": np.array([-4.8, -0.13, 0.27, 0.54]),
    "wt_init": 'none',
    "smax_temp": 0.5, # temperature for softmax, not used in normalization
    
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 0.01,
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.90,
    # seeds: 1,2,3,4,5

    "top_k": 4,
    "epochs_fine_tune": 10,
    "mix_experts": 2,
    "learning_rate_tune": 0.0001,
    "detach": False,
    "use_pretrained": False,

}
# tao 2.58, dtsemnet 2.645

config_ctslice = {
    "code": "ctslice",
    "name": "CT Slice",
    "filepath": "datasets/ctslice/",
    "n_attributes": 384,
    "attributes": [],
    "n_classes": 1,
    "classes": [],
    "use_L1": True, 
    "lamda_L1": 5e-5, 
    "nseed": 5, #5
    "tseed": None,
    "grad_clip": False,
    "over_param": [],
    "reg_hidden": 0,
    "test_split": 0.2,
    "val_split": 0.2,
    "depth": 5,
    "ste": "hard_argmax_ste", # "gumbel_softmax_ste", "gumbel_max_ste", "relu_ste", "clipped_relu_ste", "entmax_ste"][-1],
    "bins": np.array([-2.1, -1.14, -0.76, -0.49, -0.13,  0.25, 0.75,  1.4]),
    "wt_init": 'none', # xavier normal weight init ['ortho, 'xavier_uniform', 'xavier_norm', 'none']
    "smax_temp": 0.5, # temperature for softmax, not used in normalization
    
    "epochs": 40, # 40 or 70
    "batch_size": 128,
    "learning_rate": 0.001, #0.001
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.95,
    
    "top_k": 4,
    "epochs_fine_tune": 30, # 30
    "mix_experts": 2, # 2
    "learning_rate_tune": 0.0001,
    "detach": False,
    "use_pretrained": False,
}
# 0.0001: 5.04
# 0.0005: 3.43
# 0.001: 3.96
# 0.002: 3.61
# 0.003: 4.04
# 0.005: 5.04

# 3.824 with [8] overparam and 0.001 with detach-off, and 3.5831 with [0]

config_year = {
    "code": "year",
    "name": "Year Prediction",
    "filepath": "datasets/year/",
    "n_attributes": 90,
    "attributes": [],
    "n_classes": 1,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 1e-4,
    "gamma_L1": 1e-4,
    "over_param": [6],
    "reg_hidden": 0,
    "grad_clip": False,
    "nseed": 1,
    "tseed": 1,
    "test_split": None,
    "val_split": 0.2,
    "depth": 6,
    "ste": "hard_argmax_ste", # "gumbel_softmax_ste", "gumbel_max_ste", "relu_ste", "clipped_relu_ste", "entmax_ste"][-1],
    "bins": np.array([-2.1, -1.14, -0.76, -0.49, -0.13,  0.25, 0.75,  1.4]),
    "wt_init": 'none', # xavier normal weight init ['ortho, 'xavier_uniform', 'xavier_norm', 'none']
    "smax_temp": 0.2, # temperature for softmax, not used in normalization
    
    "epochs": 40,
    "batch_size": 128,
    "learning_rate": 0.001,
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.90,

    "top_k": 4,
    "epochs_fine_tune": 10,
    "mix_experts": 2,
    "learning_rate_tune": 0.0001,
    "detach": False,
    "use_pretrained": False,
}


config_ms = {
    "code": "ms",
    "name": "microsoft",
    "filepath": "datasets/ms/",
    "n_attributes": 136,
    "attributes": [],
    "n_classes": 1,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 1e-5,
    "nseed":1,
    "tseed": None,
    "grad_clip": False,
    "over_param": [],
    "reg_hidden": 0,
    "test_split": 0.2,
    "val_split": 0.2,
    "depth": 5,
    "ste": "hard_argmax_ste", # "gumbel_softmax_ste", "gumbel_max_ste", "relu_ste", "clipped_relu_ste", "entmax_ste"][-1],
    "bins": np.array([-2.1, -1.14, -0.76, -0.49, -0.13,  0.25, 0.75,  1.4]),
    "wt_init": 'none', # xavier normal weight init ['ortho, 'xavier_uniform', 'xavier_norm', 'none']
    "smax_temp": 0.5, # temperature for softmax, not used in normalization
    
    "epochs": 30,
    "batch_size": 256,
    "learning_rate": 0.002,
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.90,

    "top_k": 4,
    "epochs_fine_tune": 10,
    "mix_experts": 2,
    "learning_rate_tune": 0.0001,
    "detach": False,
    "use_pretrained": False,
}


##### NEW DATASSETS
###################
config_wine_quality = {
    "code": "wine_quality",
    "name": "wine_quality",
    "filepath": "datasets/wine_quality/",
    "n_attributes": 11,
    "attributes": [],
    "n_classes": 1,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 5e-4, 
    "over_param": [2],
    "reg_hidden": 0,
    "nseed": 1,
    "tseed": 1,
    "grad_clip": True,
    "test_split": None,
    "val_split": 0.2,
    "depth": 5,
    "ste": "hard_argmax_ste",
    "bins": np.array([0, 0.5]),
    "wt_init": 'none', # xavier normal weight init ['ortho, 'xavier_uniform', 'xavier_norm', 'none']
    "smax_temp": 0.1,

    
    "epochs": 30,
    "batch_size": 64,
    "learning_rate": 0.01, #0.01 for top1
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.90,

    "top_k": 4,
    "epochs_fine_tune": 5,
    "mix_experts": 2,
    "learning_rate_tune": 0.0001,
    "detach": False,
    "use_pretrained": False,

#X size: (6497, 11), y size: (6497,)
}


config_superconduct = {
    "code": "superconduct",
    "name": "superconduct",
    "filepath": "datasets/superconduct/",
    "n_attributes": 79,
    "attributes": [],
    "n_classes": 1,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 1e-5, 
    "over_param": [],
    "reg_hidden": 0,
    "nseed": 1,
    "tseed": 1,
    "grad_clip": True,
    "test_split": None,
    "val_split": 0.2,
    "depth": 5,
    "ste": "hard_argmax_ste",
    "bins": np.array([0, 0.5]),
    "wt_init": 'none',
    "smax_temp": 0.5,
    
    "epochs": 50,
    "batch_size": 128,
    "learning_rate": 0.01, #0.01 for top1
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.90,

    "top_k": 4,
    "epochs_fine_tune": 10,
    "mix_experts": 2,
    "learning_rate_tune": 0.0001,
    "detach": False,
    "use_pretrained": False,

#X size: (21263, 79), y size: (21263,)
#6 - 1.39
# 5 - 1.34
# 4 - 1.37
}


config_pol= {
    "code": "pol",
    "name": "pol",
    "filepath": "datasets/pol/",
    "n_attributes": 26,
    "attributes": [],
    "n_classes": 1,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 1e-5, 
    "over_param": [],
    "reg_hidden": 0,
    "nseed": 1,
    "tseed": 1,
    "grad_clip": True,
    "test_split": None,
    "val_split": 0.2,
    "depth": 5,
    "ste": "hard_argmax_ste",
    "bins": np.array([0, 0.5]),
    "wt_init": 'none',
    "smax_temp": 0.5,
    
    "epochs": 50,
    "batch_size": 128,
    "learning_rate": 0.01, #0.01 for top1
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.90,

    "top_k": 4,
    "epochs_fine_tune": 10,
    "mix_experts": 2,
    "learning_rate_tune": 0.0001,
    "detach": False,
    "use_pretrained": False,

#X size: (15000, 26), y size: (15000,)
# 5 - 6.9
# 6 - 7.0851
# 4 - 7.28
}

config_elevators= {
    "code": "elevators",
    "name": "elevators",
    "filepath": "datasets/elevators/",
    "n_attributes": 16,
    "attributes": [],
    "n_classes": 1,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 1e-4, 
    "over_param": [2],
    "reg_hidden": 0,
    "nseed": 1,
    "tseed": 1,
    "grad_clip": True,
    "test_split": None,
    "val_split": 0.2,
    "depth": 4,
    "ste": "hard_argmax_ste",
    "bins": np.array([0, 0.5]),
    "wt_init": 'none',
    "smax_temp": 0.1,
    
    "epochs": 20,
    "batch_size": 128,
    "learning_rate": 0.01, 
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.90,

    "top_k": 2,
    "epochs_fine_tune": 0,
    "mix_experts": 1,
    "learning_rate_tune": 0.0001,
    "detach": False,
    "use_pretrained": False,

#X size: (16599, 16), y size: (16599,)
#5 0.2237
#6 0.2199
#4 0.2167
#3 0.2243
}

config_Bike_Sharing_Demand= {
    "code": "Bike_Sharing_Demand",
    "name": "Bike_Sharing_Demand",
    "filepath": "datasets/Bike_Sharing_Demand/",
    "n_attributes": 6,
    "attributes": [],
    "n_classes": 1,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 1e-4, 
    "over_param": [],
    "reg_hidden": 0,
    "nseed": 1,
    "tseed": 1,
    "grad_clip": True,
    "test_split": None,
    "val_split": 0.2,
    "depth": 6,
    "ste": "hard_argmax_ste",
    "bins": np.array([0, 0.5]),
    "wt_init": 'none',
    "smax_temp": 0.5,
    
    "epochs": 40,
    "batch_size": 128,
    "learning_rate": 0.01, #0.01 for top1
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.95,

    "top_k": 4,
    "epochs_fine_tune": 10,
    "mix_experts": 2,
    "learning_rate_tune": 0.0001,
    "detach": False,
    "use_pretrained": False,

#X size: (17379, 6), y size: (17379,)
#5 1.1899
#4 1.2061
#6 1.1522
#7 1.1594
}

config_sulfur= {
    "code": "sulfur",
    "name": "sulfur",
    "filepath": "datasets/sulfur/",
    "n_attributes": 6,
    "attributes": [],
    "n_classes": 1,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 5e-5, 
    "over_param": [],
    "reg_hidden": 0,
    "nseed": 1,
    "tseed": 1,
    "grad_clip": True,
    "test_split": None,
    "val_split": 0.2,
    "depth": 5,
    "ste": "hard_argmax_ste",
    "bins": np.array([0, 0.5]),
    "wt_init": 'none',
    "smax_temp": 0.5,
    
    "epochs": 20,
    "batch_size": 128,
    "learning_rate": 0.01, #0.01 for top1
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.95,

    "top_k": 4,
    "epochs_fine_tune": 10,
    "mix_experts": 2,
    "learning_rate_tune": 0.0001,
    "detach": False,
    "use_pretrained": False,

#X size: (10081, 6), y size: (10081,)
# 5 0.3261
# 6 0.3616
# 4 0.3566
}


### Large New Datasets
config_medical = {
    "code": "medical",
    "name": "medical",
    "filepath": "datasets/medical/",
    "n_attributes": 3,
    "attributes": [],
    "n_classes": 1,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 1e-5, 
    "over_param": [],
    "reg_hidden": 0,
    "nseed": 1,
    "tseed": 1,
    "grad_clip": True,
    "test_split": None,
    "val_split": 0.2,
    "depth": 5,
    "ste": "hard_argmax_ste",
    "bins": np.array([0, 0.5]),
    "wt_init": 'none',
    "smax_temp": 0.5,
    
    "epochs": 15,
    "batch_size": 128,
    "learning_rate": 0.001, #0.01 for top1
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.95,

    "top_k": 4,
    "epochs_fine_tune": 5,
    "mix_experts": 2,
    "learning_rate_tune": 0.0001,
    "detach": False,
    "use_pretrained": False,

# X size: (163065, 3), y size: (163065,) 
# 5 0.8346
# 6 0.8520
# 4 0.8433
}



config_houses= {
    "code": "houses",
    "name": "houses",
    "filepath": "datasets/houses/",
    "n_attributes": 8,
    "attributes": [],
    "n_classes": 1,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 1e-5, 
    "over_param": [],
    "reg_hidden": 0,
    "nseed": 1,
    "tseed": 1,
    "grad_clip": True,
    "test_split": None,
    "val_split": 0.2,
    "depth": 5,
    "ste": "hard_argmax_ste",
    "bins": np.array([0, 0.5]),
    "wt_init": 'none',
    "smax_temp": 0.5,
    
    "epochs": 20,
    "batch_size": 64,
    "learning_rate": 0.01, 
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.95,

    "top_k": 4,
    "epochs_fine_tune": 10,
    "mix_experts": 1,
    "learning_rate_tune": 0.0001,
    "detach": False,
    "use_pretrained": False,

#X size: (581835, 9), y size: (581835,)
}


config_taxi= {
    "code": "taxi",
    "name": "taxi",
    "filepath": "datasets/nyc_taxi_green/",
    "n_attributes": 9,
    "attributes": [],
    "n_classes": 1,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 1e-5, 
    "over_param": [],
    "reg_hidden": 0,
    "nseed": 1,
    "tseed": 1,
    "grad_clip": True,
    "test_split": None,
    "val_split": 0.2,
    "depth": 6,
    "ste": "hard_argmax_ste",
    "bins": np.array([0, 0.5]),
    "wt_init": 'none',
    "smax_temp": 0.5,
    
    "epochs": 20,
    "batch_size": 128,
    "learning_rate": 0.01, #0.01 for top1
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.95,

    "top_k": 4,
    "epochs_fine_tune": 10,
    "mix_experts": 2,
    "learning_rate_tune": 0.0001,
    "detach": False,
    "use_pretrained": False,

#X size: (581835, 9), y size: (581835,)
}




### Synthetic Datasets
config_synth_5k= {
    "code": "synth_5k",
    "name": "synth_5k",
    "filepath": "datasets/synthetic/",
    "n_attributes": 4,
    "attributes": [],
    "n_classes": 1,
    "classes": [],
    "use_L1": False,
    "lamda_L1": 1e-5, 
    "over_param": [],
    "reg_hidden": 0,
    "nseed": 1,
    "tseed": 1,
    "grad_clip": True,
    "test_split": None,
    "val_split": None,
    "depth": 5,
    "ste": "hard_argmax_ste",
    "bins": np.array([0, 0.5]),
    "wt_init": 'none',
    "smax_temp": 0.1,
    
    "epochs": 15,
    "batch_size": 128,
    "learning_rate": 0.01, #0.01 for top1
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.95,

    "top_k": 4,
    "epochs_fine_tune": 10,
    "mix_experts": 2,
    "learning_rate_tune": 0.0005,
    "detach": False,
    "use_pretrained": False,

#X size: (581835, 9), y size: (581835,)
}


config_synth_30k= {
    "code": "synth_30k",
    "name": "synth_30k",
    "filepath": "datasets/synthetic/",
    "n_attributes": 4,
    "attributes": [],
    "n_classes": 1,
    "classes": [],
    "use_L1": False,
    "lamda_L1": 1e-5, 
    "over_param": [],
    "reg_hidden": 0,
    "nseed": 1,
    "tseed": 1,
    "grad_clip": True,
    "test_split": None,
    "val_split": None,
    "depth": 5,
    "ste": "hard_argmax_ste",
    "bins": np.array([0, 0.5]),
    "wt_init": 'none',
    "smax_temp": 0.1,
    
    "epochs": 15,
    "batch_size": 128,
    "learning_rate": 0.01, #0.01 for top1
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.95,

    "top_k": 4,
    "epochs_fine_tune": 10,
    "mix_experts": 1,
    "learning_rate_tune": 0.0001,
    "detach": False,
    "use_pretrained": False,

#X size: (581835, 9), y size: (581835,)
}

config_synth_5l= {
    "code": "synth_5l",
    "name": "synth_5l",
    "filepath": "datasets/synthetic/",
    "n_attributes": 4,
    "attributes": [],
    "n_classes": 1,
    "classes": [],
    "use_L1": False,
    "lamda_L1": 1e-5, 
    "over_param": [],
    "reg_hidden": 0,
    "nseed": 1,
    "tseed": 1,
    "grad_clip": True,
    "test_split": None,
    "val_split": None,
    "depth": 5,
    "ste": "hard_argmax_ste",
    "bins": np.array([0, 0.5]),
    "wt_init": 'none',
    "smax_temp": 0.1,
    
    "epochs": 15,
    "batch_size": 128,
    "learning_rate": 0.01, #0.01 for top1
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.95,

    "top_k": 4,
    "epochs_fine_tune": 10,
    "mix_experts": 2,
    "learning_rate_tune": 0.0005,
    "detach": False,
    "use_pretrained": False,

#X size: (581835, 9), y size: (581835,)
}

