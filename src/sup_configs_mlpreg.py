import sys
sys.path.append(".")

import csv
import pdb

import numpy as np
import pandas as pd

config_AB = {
    "code": "abalone",
    "name": "Abalone",
    "filepath": "datasets/abalone.data",
    "n_attributes": 10,
    "attributes": ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings'],
    "n_classes": 1,
    "classes": [],
    "test_split": 0.4,
    "val_split": 0.2,
    "depth": 6,
    
    "epochs": 200,
    "batch_size": 32,
    "learning_rate": 0.05,
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.95

    # "epochs": 100,
    # "batch_size": 64,
    # "learning_rate": 0.08,
    # "lr_scheduler": True,
    # "lr_scheduler_gamma": 0.98
}

config_AL = {
    "code": "ailerons",
    "name": "Ailerons",
    "filepath": "datasets/ailerons/",
    "n_attributes": 40,
    "attributes": ['climbRate', 'Sgz', 'p', 'q', 'curPitch', 'curRoll', 'absRoll', 'diffClb', 'diffRollRate', 'diffDiffClb', 'SeTime1', 'SeTime2', 'SeTime3', 'SeTime4', 'SeTime5', 'SeTime6', 'SeTime7', 'SeTime8', 'SeTime9', 'SeTime10', 'SeTime11', 'SeTime12', 'SeTime13', 'SeTime14', 'diffSeTime1', 'diffSeTime2', 'diffSeTime3', 'diffSeTime4', 'diffSeTime5', 'diffSeTime6', 'diffSeTime7', 'diffSeTime8', 'diffSeTime9', 'diffSeTime10', 'diffSeTime11', 'diffSeTime12', 'diffSeTime13', 'diffSeTime14', 'alpha', 'Se', 'goal'],
    "n_classes": 1,
    "classes": [],
    "test_split": None,
    "val_split": 0.2,
    "depth": 6,
    
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 0.002,
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.98
}

real_dataset_list = ["abalone", "ailerons"]

def get_config(dataset_code):
    for config in [config_AB, config_AL]:

        if config["code"] == dataset_code:
            return config

    raise Exception(f"Invalid dataset code {dataset_code}.")


def load_dataset(config):
    if config["code"] == "abalone":
        #==== Convert categorical to one hot
        abalone_data = pd.read_csv(config["filepath"], names=config["attributes"])
        # Convert categorical variable 'Sex' into dummy/indicator variables
        # abalone_data = pd.get_dummies(abalone_data, columns=['Sex'])
        abalone_data = pd.get_dummies(abalone_data, columns=[abalone_data.columns[0]])
        # abalone_data.drop('Sex_Sex', axis=1, inplace=True)
        # Split dataset into features and target
        X = abalone_data.drop('Rings', axis=1).values #[1:]  # Remove head from X
        y = abalone_data['Rings'].values #[1:]  # Remove head from y
        # Convert to integer
        
        X = X.astype(float)
        y = y.astype(float)
        return X, y
    
    elif config["code"] == "ailerons":
        #==== Convert categorical to one hot
        ailerons_data = pd.read_csv(f"{config['filepath']}ailerons.data", names=config["attributes"])
        ailerons_test = pd.read_csv(f"{config['filepath']}ailerons.test", names=config["attributes"])
        
        X = ailerons_data.drop('goal', axis=1).values 
        y = ailerons_data['goal'].values * 1e4

        X_test = ailerons_test.drop('goal', axis=1).values 
        y_test = ailerons_test['goal'].values * 1e4
        # Convert to integer
        
        X = X.astype(float)
        y = y.astype(float)
        X_test = X_test.astype(float)
        y_test = y_test.astype(float)
        
        return X, y, X_test, y_test
        


if __name__ == "__main__":
    pdb.set_trace()