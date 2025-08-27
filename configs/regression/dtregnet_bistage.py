import numpy as np
config_AB = {
    "code": "abalone",
    "name": "Abalone",
    "filepath": "datasets/abalone.data",
    "n_attributes": 10,
    "attributes": ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings'],
    "n_classes": 1,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 5e-4, # 5e-4
    "grad_clip": True,
    "over_param": [8],
    "reg_hidden": 0,
    "nseed": 1,
    "tseed": 1,
    "test_split": 0.4,
    "val_split": 0.2,
    "depth": 5,
    "bins": [np.array([-2.82, -0.607, -0.2952, 0.3284])], # np.array([-2.82, -0.2951,])
    
    # [np.array([-2.82, -0.607, -0.2952, 0.3284]), 
    #          np.array([-2.82, -0.61, -0.295, 0.33]), 
             
    #          np.array([-2.82, -0.62, -0.1, 0.31]), 
             
    #          np.array([-2.82, -0.61, -0.02, 0.34]),
    #          np.array([-2.82, -0.59, -0.28, 0.35]),
    #         ],
    
    "epochs_s1": 20,
    "epochs_s2": 50,
    "batch_size": 32,
    "learning_rate_s1": 0.02, # 0.02
    "learning_rate_s2": 0.001, # 0.05
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.90 # 0.90


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
    "lamda_L1": 5e-4,
    "over_param": [],
    "reg_hidden": 0, #16
    "nseed": 1,
    "tseed": 1,
    "grad_clip": True,
    "test_split": None,
    "val_split": 0.2,
    "depth": 5,
    "bins": np.array([-7, -0.5695, 0.1649, 0.655]),  
    
    
    "epochs_s1": 20,
    "epochs_s2": 20,
    "batch_size": 64,
    "learning_rate_s1": 0.05,
    "learning_rate_s2": 0.01,
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.90
}

config_PB = {
    "code": "pdb_bind",
    "name": "PDB Bind",
    "filepath": "datasets/pdb_bind/",
    "n_attributes": 2052,
    "attributes": [],
    "n_classes": 1,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 2e-2,
    "nseed": 1,
    "over_param": [],
    "reg_hidden": 0,
    "tseed": None,
    "grad_clip": False,
    "test_split": None,
    "val_split": None,
    "depth": 2,
    "bins": np.array([-3.1, -0.0]),
    # np.array([-3.1, 0.03])
    # np.array([-3.1, -0.72, 0.037, 0.72])
    # np.array([-3.1, -0.455, 0.495])
    
    "epochs_s1": 15,
    "epochs_s2": 15,
    "batch_size": 256,
    "learning_rate_s1": 0.001,
    "learning_rate_s2": 0.0005,
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.95
}

config_YR = {
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
    "reg_hidden": 16,
    "grad_clip": False,
    "nseed": 1,
    "tseed": 1,
    "test_split": None,
    "val_split": 0.2,
    "depth": 6,
    "bins": np.array([-7, -0.401, 0.33, 0.696]),
    # np.array([-7, -1.133, -0.40, 0.056, 0.33, 0.513, 0.696, 0.879]) ,
    # np.array([-7, -0.401, 0.33, 0.696])
    
    "epochs_s1": 40,
    "epochs_s2": 60,
    "batch_size": 128,
    "learning_rate_s1": 0.001,
    "learning_rate_s2": 0.001,
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.95
}

config_CA = {
    "code": "cpu_active",
    "name": "CPU Active",
    "filepath": "datasets/cpu_active/",
    "n_attributes": 21,
    "attributes": ['time', 'lread', 'lwrite', 'scall', 'sread', 'swrite', 'fork', 'exec', 'rchar', 'wchar', 'pgout', 'ppgout', 'pgfree', 'pgscan', 'atch', 'pgin', 'ppgin', 'pflt', 'vflt', 'runqsz', 'runocc', 'freemem', 'freeswap', 'usr', 'sys', 'wio', 'idle'],
    "n_classes": 0,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 5e-4,
    "over_param": [6],
    "reg_hidden": 0, #16
    "nseed": 5,
    "tseed": 1,
    "grad_clip": True,
    "test_split": 0.4,
    "val_split": 0.2,
    "depth": 5,
    "bins": np.array([-4.8, -0.13, 0.27, 0.54]),
    # np.array([-4.4, -0.5, -0.14, 0.12, 0.27,  0.38, 0.54,  0.64])
    #np.array([-4.4, -0.14, 0.279, 0.54])
    
    "epochs_s1": 30,
    "epochs_s2": 50,
    "batch_size": 128,
    "learning_rate_s1": 0.01,
    "learning_rate_s2": 0.01, 
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.95
}

config_ctslice = {
    "code": "ctslice",
    "name": "CT Slice",
    "filepath": "datasets/ctslice/",
    "n_attributes": 384,
    "attributes": [],
    "n_classes": 1,
    "classes": [],
    "use_L1": True,
    "lamda_L1":5e-6,
    "nseed":5,
    "tseed": None,
    "grad_clip": True,
    "over_param": [],
    "reg_hidden": 0,
    "test_split": 0.2,
    "val_split": 0.2,
    "depth": 5,
    "bins": np.array([-2.1, -1.14, -0.76, -0.49, -0.13,  0.25, 0.75,  1.4]),
    
    "epochs_s1": 25,
    "epochs_s2": 10,
    "batch_size": 128,
    "learning_rate_s1": 0.005,
    "learning_rate_s2": 0.01, 
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.90
 
    # t9: 0.01, 0.02, 5e-6, 0.90
}

config_MS = {
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
    "grad_clip": True,
    "over_param": [],
    "reg_hidden": 64,
    "test_split": 0.2,
    "val_split": 0.2,
    "depth": 5,
    "bins": np.array([-1, 0, 1, 2]), 
    
    "epochs_s1": 50,
    "epochs_s2": 80,
    "batch_size": 256,
    "learning_rate_s1": 0.001,
    "learning_rate_s2": 0.01,
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.90
}

