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
    "lamda_L1": 5e-4,
    "grad_clip": False,
    "over_param": [8],
    "reg_hidden": 16,
    "nseed": 5,
    "tseed": 1,
    "test_split": 0.4,
    "val_split": 0.2,
    "depth": 5,
    "ste": ["hard_argmax_ste", "gumbel_softmax_ste", "gumbel_max_ste", "relu_ste", "clipped_relu_ste", "entmax_ste"][-1],
    
    "epochs": 80,
    "batch_size": 64,
    "learning_rate": 0.01,
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.98
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
    "lamda_L1": 1e-4,
    "over_param": [],
    "reg_hidden": 0,
    "nseed": 1,
    "tseed": 1,
    "grad_clip": True,
    "test_split": None,
    "val_split": 0.3,
    "depth": 5,
    "ste": ["hard_argmax_ste", "gumbel_softmax_ste", "gumbel_max_ste", "relu_ste", "clipped_relu_ste", "entmax_ste"][-1],
    "bins": np.array([-7, -0.5695, 0.1649, 0.655]),
    
    "epochs": 50,
    "batch_size": 128,
    "learning_rate": 0.01,
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.90

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
    "lamda_L1": 1e-2,
    "nseed": 1,
    "over_param": [],
    "reg_hidden": 0,
    "tseed": None,
    "grad_clip": False,
    "test_split": None,
    "val_split": None,
    "depth": 2,
    
    "epochs": 100,
    "batch_size": 256,
    "learning_rate": 0.01,
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.95
}

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
    "reg_hidden": 32,
    "grad_clip": False,
    "nseed": 1,
    "tseed": 1,
    "test_split": None,
    "val_split": 0.2,
    "depth": 6,
    
    "epochs": 40,
    "batch_size": 128,
    "learning_rate": 0.001,
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.90
}

config_cpu_active = {
    "code": "cpu_active",
    "name": "CPU Active",
    "filepath": "datasets/cpu_active/",
    "n_attributes": 21,
    "attributes": ['time', 'lread', 'lwrite', 'scall', 'sread', 'swrite', 'fork', 'exec', 'rchar', 'wchar', 'pgout', 'ppgout', 'pgfree', 'pgscan', 'atch', 'pgin', 'ppgin', 'pflt', 'vflt', 'runqsz', 'runocc', 'freemem', 'freeswap', 'usr', 'sys', 'wio', 'idle'],
    "n_classes": 0,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 1e-4,
    "over_param": [6],
    "reg_hidden": 0,
    "nseed": 1,
    "tseed": 1,
    "grad_clip": False,
    "test_split": 0.4,
    "val_split": 0.2,
    "depth": 5,
    "ste": ["hard_argmax_ste", "gumbel_softmax_ste", "gumbel_max_ste", "relu_ste", "clipped_relu_ste", "entmax_ste"][-1],
    "bins": np.array([-4.8, -0.13, 0.27, 0.54]),

    "epochs": 100,
    "batch_size": 128,
    "learning_rate": 0.02, # 0.05
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.90 # 0.90
    # seeds: 1,2,3,4,5
}

config_ctslice = {
    "code": "ctslice",
    "name": "CT Slice",
    "filepath": "datasets/ctslice/",
    "n_attributes": 384,
    "attributes": [],
    "n_classes": 1,
    "classes": [],
    "use_L1": False,
    "lamda_L1": 1e-5,
    "nseed":5,
    "tseed": None,
    "grad_clip": False,
    "over_param": [32, 16],
    "reg_hidden": 64,
    "test_split": 0.2,
    "val_split": 0.2,
    "depth": 5,
    "ste": ["hard_argmax_ste", "gumbel_softmax_ste", "gumbel_max_ste", "relu_ste", "clipped_relu_ste", "entmax_ste"][-1],
    "bins": np.array([-2.1, -1.14, -0.76, -0.49, -0.13,  0.25, 0.75,  1.4]),
    
    "epochs": 50,
    "batch_size": 128,
    "learning_rate": 0.001,
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.90
    
}

config_ms = {
    "code": "ms",
    "name": "microsoft",
    "filepath": "datasets/ms/",
    "n_attributes": 136,
    "attributes": [],
    "n_classes": 1,
    "classes": [],
    "use_L1": False,
    "lamda_L1": 2e-5,
    "nseed":1,
    "tseed": None,
    "grad_clip": False,
    "over_param": [],
    "reg_hidden": 64,
    "test_split": 0.2,
    "val_split": 0.2,
    "depth": 5,
    
    "epochs": 30,
    "batch_size": 256,
    "learning_rate": 0.002,
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.90
}

config_yahoo = {
    "code": "yahoo",
    "name": "yahoo",
    "filepath": "datasets/yahoo/",
    "n_attributes": 136,
    "attributes": [],
    "n_classes": 1,
    "classes": [],
    "use_L1": False,
    "lamda_L1": 1e-5,
    "nseed":1,
    "tseed": None,
    "grad_clip": False,
    "over_param": [],
    "reg_hidden": 64,
    "test_split": 0.2,
    "val_split": 0.2,
    "depth": 5,
    
    "epochs": 20,
    "batch_size": 256,
    "learning_rate": 0.001,
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.90
}


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
    
    "epochs": 60,
    "batch_size": 64,
    "learning_rate": 0.01, #0.01 for top1
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.90,

     "use_pretrained": False

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
    
    "epochs": 60,
    "batch_size": 128,
    "learning_rate": 0.01, #0.01 for top1
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.90

#X size: (21263, 79), y size: (21263,)
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
    
    "epochs": 50,
    "batch_size": 128,
    "learning_rate": 0.01, #0.01 for top1
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.90

#X size: (15000, 26), y size: (15000,)
}

config_elevators= {
    "code": "elevators",
    "name": "elevators",
    "filepath": "datasets/elevators/",
    "n_attributes": 16,
    "attributes": [],
    "n_classes": 1,
    "classes": [],
    "use_L1": False,
    "lamda_L1": 5e-4, 
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
    
    "epochs": 20,
    "batch_size": 128,
    "learning_rate": 0.01, #0.01 for top1
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.90

#X size: (16599, 16), y size: (16599,)
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
    "depth": 5,
    "ste": "hard_argmax_ste",
    "bins": np.array([0, 0.5]),
    
    "epochs": 60,
    "batch_size": 128,
    "learning_rate": 0.01, #0.01 for top1
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.90

#X size: (17379, 6), y size: (17379,)
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
    
    "epochs": 30,
    "batch_size": 128,
    "learning_rate": 0.05, #0.01 for top1
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.95

#X size: (10081, 6), y size: (10081,)
}


config_medical = {
    "code": "medical",
    "name": "medical",
    "filepath": "datasets/medical/",
    "n_attributes": 3,
    "attributes": [],
    "n_classes": 1,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 5e-4, 
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
    
    "epochs": 20,
    "batch_size": 128,
    "learning_rate": 0.005, #0.01 for top1
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.95

# X size: (163065, 3), y size: (163065,) 
}

config_nyc_taxi_green= {
    "code": "nyc_taxi_green",
    "name": "nyc_taxi_green",
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
    "depth": 5,
    "ste": "hard_argmax_ste",
    "bins": np.array([0, 0.5]),
    
    "epochs": 40,
    "batch_size": 128,
    "learning_rate": 0.01, #0.01 for top1
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.95,

   

#X size: (581835, 9), y size: (581835,)
}