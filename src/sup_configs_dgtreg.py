import sys
sys.path.append(".")

import csv
import pdb

import numpy as np
import pandas as pd
import gzip
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

config_AB = {
    "code": "abalone",
    "name": "Abalone",
    "filepath": "datasets/abalone.data",
    "n_attributes": 10,
    "attributes": ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings'],
    "n_classes": 1,
    "classes": [],
    "tseed": 1,
    "test_split": 0.4,
    "val_split": 0.2,
    "depth": 6,
    
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.1,
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.94

}

config_AL = {
    "code": "ailerons",
    "name": "Ailerons",
    "filepath": "datasets/ailerons/",
    "n_attributes": 40,
    "attributes": ['climbRate', 'Sgz', 'p', 'q', 'curPitch', 'curRoll', 'absRoll', 'diffClb', 'diffRollRate', 'diffDiffClb', 'SeTime1', 'SeTime2', 'SeTime3', 'SeTime4', 'SeTime5', 'SeTime6', 'SeTime7', 'SeTime8', 'SeTime9', 'SeTime10', 'SeTime11', 'SeTime12', 'SeTime13', 'SeTime14', 'diffSeTime1', 'diffSeTime2', 'diffSeTime3', 'diffSeTime4', 'diffSeTime5', 'diffSeTime6', 'diffSeTime7', 'diffSeTime8', 'diffSeTime9', 'diffSeTime10', 'diffSeTime11', 'diffSeTime12', 'diffSeTime13', 'diffSeTime14', 'alpha', 'Se', 'goal'],
    "n_classes": 1,
    "classes": [],
    "tseed": 1,
    "test_split": None,
    "val_split": 0.2,
    "depth": 6,
    
    "epochs": 60,
    "batch_size": 64,
    "learning_rate": 0.04,
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.9
}

config_CA = {
    "code": "cpu_active",
    "name": "CPU Active",
    "filepath": "datasets/cpu_active/",
    "n_attributes": 21,
    "attributes": ['time', 'lread', 'lwrite', 'scall', 'sread', 'swrite', 'fork', 'exec', 'rchar', 'wchar', 'pgout', 'ppgout', 'pgfree', 'pgscan', 'atch', 'pgin', 'ppgin', 'pflt', 'vflt', 'runqsz', 'runocc', 'freemem', 'freeswap', 'usr', 'sys', 'wio', 'idle'],
    "n_classes": 0,
    "classes": [],
    "tseed": 1,
    "test_split": 0.4,
    "val_split": 0.2,
    "depth": 6,
    
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 0.2,
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.98
}

config_PB = {
    "code": "pdb_bind",
    "name": "PDB Bind",
    "filepath": "datasets/pdb_bind/",
    "n_attributes": 2052,
    "attributes": [],
    "n_classes": 1,
    "classes": [],
    "tseed": 1,
    "test_split": None,
    "val_split": None,
    "depth": 6,
    
    "epochs": 80,
    "batch_size": 128,
    "learning_rate": 0.4,
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
    "tseed": 1,
    "test_split": None,
    "val_split": 0.2,
    "depth": 8,
    
    "epochs": 50,
    "batch_size": 128,
    "learning_rate": 0.2,
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.9
}

config_CS = {
    "code": "ctslice",
    "name": "CT Slice",
    "filepath": "datasets/ctslice/",
    "n_attributes": 384,
    "attributes": [],
    "n_classes": 1,
    "classes": [],
    "tseed": 1,
    "test_split": None,
    "val_split": 0.2,
    "depth": 10,
    
    "epochs": 60,
    "batch_size": 128,
    "learning_rate": 0.2,
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.95
}

real_dataset_list = ["abalone", "ailerons", "cpu_active", "pdb_bind", "year", "ctslice"]


def get_config(dataset_code):
    for config in [config_AB, config_AL, config_CA, config_PB, config_YR, config_CS]:

        if config["code"] == dataset_code:
            return config

    raise Exception(f"Invalid dataset code {dataset_code}.")


def load_dataset(config, standardize=True):
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
        X, X_test, y, y_test = train_test_split(X, y, test_size=config["test_split"], random_state=config["tseed"])#data_config["tseed"])    
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config["val_split"], random_state=config["tseed"])

        

    
    elif config["code"] == "ailerons":
        #==== Convert categorical to one hot
        ailerons_data = pd.read_csv(f"{config['filepath']}ailerons.data", names=config["attributes"])
        ailerons_test = pd.read_csv(f"{config['filepath']}ailerons.test", names=config["attributes"])
        
        X = np.array(ailerons_data.iloc[:, :-1], dtype=np.float32)
        y = np.array(ailerons_data.iloc[:, -1], dtype=np.float32) * 1e4 #info: scale y to be in the same range as x

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config["val_split"], random_state=config["tseed"])

        X_test = np.array(ailerons_test.iloc[:, :-1], dtype=np.float32)
        y_test = np.array(ailerons_test.iloc[:, -1], dtype=np.float32) * 1e4
        
    
    elif config["code"] == "cpu_active":
        cactv_data = pd.read_csv(f"{config['filepath']}comp.data", header=None, sep=' ')
        
        X = cactv_data.iloc[:, 1:22].values
        y = cactv_data.iloc[:, 23].values

        X = X.astype(float)
        y = y.astype(float)

        X, X_test, y, y_test = train_test_split(X, y, test_size=config["test_split"], random_state=config["tseed"])#data_config["tseed"])    
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config["val_split"], random_state=config["tseed"])
        

    elif config["code"] == "pdb_bind":
        with gzip.open(f"{config['filepath']}PDBbind.pkl.gz", 'rb') as f:
            X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(f)
            X_train = X_train.astype(np.float32)
            y_train = y_train.astype(np.float32)
            X_val = X_val.astype(np.float32)
            y_val = y_val.astype(np.float32)
            X_test = X_test.astype(np.float32)
            y_test = y_test.astype(np.float32)
    
    elif config["code"] == "year":
        X = np.load(f"{config['filepath']}year-train-x.npy").astype(np.float32)
        y = np.load(f"{config['filepath']}year-train-y.npy").astype(np.float32)
        X_test = np.load(f"{config['filepath']}year-test-x.npy").astype(np.float32)
        y_test = np.load(f"{config['filepath']}year-test-y.npy").astype(np.float32)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config["val_split"], random_state=config["tseed"])
    
    elif config["code"] == "ctslice":
        df = pd.read_csv(f'{config["filepath"]}/slice_localization_data.csv')
        split_at = int(len(df) * 0.8)  # Split data into 90% train and 10% test
        
        train = df.iloc[:split_at]
        test = df.iloc[split_at:]

        train = train.iloc[:, 1:]  # Remove the first column
        test = test.iloc[:, 1:]  # Remove the first column

        X = train.iloc[:, :-1].to_numpy(np.float32)
        y = train.iloc[:, -1].to_numpy(np.int64)
        X_test = test.iloc[:, :-1].to_numpy(np.float32)
        y_test = test.iloc[:, -1].to_numpy(np.int64)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config["val_split"], random_state=config["tseed"])
        
    
    if standardize:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    pdb.set_trace()