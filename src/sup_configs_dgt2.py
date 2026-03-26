import sys
sys.path.append(".")

import csv
import pdb

import numpy as np
import pandas as pd
import gzip
import bz2
import pickle
import tarfile
import zipfile

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

import torch
import torchvision
from torchvision import transforms

# done
config_MN = {
    "code": "mnist",
    "name": "MNIST",
    "filepath": "datasets/mnist",
    "n_attributes": 784,
    "attributes": [],
    "n_classes": 10,
    "classes": [],
    "use_L1": False,
    "lamda_L1": 1e-5,
    "opt": "sgd",
    "momentum": 0.9,
    "over_param":[],
    "grad_clip": False,
    "wt_init": True,
    "tseed": 1,
    "test_split": None,
    "val_split": None,
    "preprocess": True,
    "depth": 8,
    
    "epochs": 100,
    "batch_size": 128,
    "learning_rate": 0.4, #0.2: 95.7 #0.4: 96.1
    "lr_scheduler": True,
    "lr_scheduler_type": 'linear',
    "lr_scheduler_gamma": 0.95

}

#done
config_LT = {
    "code": "letter",
    "name": "LETTER",
    "filepath": "datasets/letter/",
    "n_attributes": 16,
    "attributes": [],
    "n_classes": 26,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 5e-6,
    "opt": "rmsprop",
    "momentum": 0,
    "over_param":[6,6],
    "grad_clip": True,
    "wt_init": True,
    "tseed": 1,
    "test_split": None,
    "val_split": None,
    "preprocess": True,
    "depth": 10,
    
    "epochs": 400,
    "batch_size": 128,
    "learning_rate": 0.01, 
    "lr_scheduler": True,
    "lr_scheduler_type": 'linear',
    "lr_scheduler_gamma": 0.95
}

#done
config_SG = {
    "code": "segment",
    "name": "segment",
    "filepath": "datasets/segment/",
    "n_attributes": 19,
    "attributes": [],
    "n_classes": 7,
    "classes": [],
    "use_L1": False,
    "lamda_L1": 1e-4,
    "opt": "adam",
    "momentum": 0,
    "over_param":[6],
    "wt_init": False,
    "grad_clip": False,
    "tseed": 1,
    "test_split": 0.2,
    "val_split": 0.2,
    "preprocess": True,
    "depth": 8,
    
    "epochs": 50,
    "batch_size": 128,
    "learning_rate": 0.01,
    "lr_scheduler_type": 'linear',
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.95
}


config_C4 = {
    "code": "connect",
    "name": "Connect4",
    "filepath": "datasets/connect-4/",
    "n_attributes": 126,
    "attributes": [],
    "n_classes": 3,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 1e-4,
    "opt": "rmsprop",
    "momentum": 0,
    "over_param":[32],
    "wt_init": True,
    "grad_clip": True,
    "tseed": 1,
    "test_split": 0.2,
    "val_split": 0.2,
    "preprocess": True,
    "depth": 8,
    
    "epochs": 100,
    "batch_size": 128,
    "learning_rate": 0.01,
    "lr_scheduler_type": 'linear',
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.95
}

# done
config_SI = {
    "code": "satimages",
    "name": "satimages",
    "filepath": "datasets/satimages/",
    "n_attributes": 36,
    "attributes": [],
    "n_classes": 6,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 5e-5,
    "opt": "rmsprop",
    "momentum": 0.2,
    "over_param":[64],
    "wt_init": False,
    "grad_clip": True,
    "tseed": 1,
    "test_split": 0.2,
    "val_split": 0.2,
    "preprocess": True,
    "depth": 6,
    
    "epochs":200,
    "batch_size": 128,
    "learning_rate": 0.01,
    "lr_scheduler_type": 'linear',
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.98
}

config_CN = {
    "code": "census",
    "name": "Census1990",
    "filepath": "datasets/census/",
    "n_attributes": 67,
    "attributes": [],
    "n_classes": 18,
    "classes": [],
    "use_L1": False,
    "lamda_L1": 1e-2,
    "opt": "adam",
    "momentum": 0.1,
    "over_param":[],
    "wt_init": False,
    "grad_clip": False,
    "tseed": 1,
    "test_split": 0.2,
    "val_split": 0.2,
    "preprocess": True,
    "depth": 8,
    
    "epochs": 80,
    "batch_size": 256,
    "learning_rate": 0.001,
    "lr_scheduler": False,
    "lr_scheduler_gamma": 0.98
    
}

config_FC = {
    "code": "forest",
    "name": "ForestCover",
    "filepath": "datasets/forest/",
    "n_attributes": 54,
    "attributes": ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Cover_Type'],
    "n_classes": 7,
    "classes": [],
    "use_L1": False,
    "lamda_L1": 1e-7,
    "opt": "adam",
    "momentum": 0.1,
    "over_param":[],
    "wt_init": False,
    "grad_clip": False,
    "tseed": 1,
    "test_split": 0.2,
    "val_split": 0.2,
    "preprocess": True,
    "depth": 10,
    
    "epochs": 200,
    "batch_size": 128,
    "learning_rate": 0.01,
    "lr_scheduler": False,
    "lr_scheduler_type": 'cosine',
    "lr_scheduler_gamma": 0.95
}

config_PD = {
    "code": "pendigits",
    "name": "pendigits",
    "filepath": "datasets/pendigits/",
    "n_attributes": 16,
    "attributes": [],
    "n_classes": 10,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 1e-5,
    "opt": "rmsprop",
    "momentum": 0,
    "over_param":[64],
    "wt_init": False,
    "grad_clip": True,
    "tseed": 1,
    "test_split": 0.2,
    "val_split": 0.2,
    "preprocess": True,
    "depth": 8,
    
    "epochs":100,
    "batch_size": 128,
    "learning_rate": 0.01,
    "lr_scheduler_type": 'linear',
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.98
}

config_PR = {
    "code": "protein",
    "name": "protein",
    "filepath": "datasets/protein/",
    "n_attributes": 357,
    "attributes": [],
    "n_classes": 3,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 1e-3,
    "opt": "rmsprop",
    "momentum": 0,
    "over_param":[12,12],
    "wt_init": False,
    "grad_clip": True,
    "tseed": 1,
    "test_split": 0.2,
    "val_split": 0.2,
    "preprocess": True,
    "depth": 4,
    
    "epochs":50,
    "batch_size": 128,
    "learning_rate": 0.01,
    "lr_scheduler_type": 'linear',
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.98
}

config_IT = {
    "code": "sensit",
    "name": "SensIT",
    "filepath": "datasets/sensit/",
    "n_attributes": 100,
    "attributes": [],
    "n_classes": 3,
    "classes": [],
    "use_L1": True,
    "lamda_L1": 1e-4,
    "opt": "rmsprop",
    "momentum": 0,
    "over_param":[6], # [6] gives 84.29 (reported) and [4] gives 84.23
    "wt_init": True,
    "grad_clip": True,
    "tseed": 1,
    "test_split": 0.2,
    "val_split": 0.2,
    "preprocess": True,
    "depth": 10,
    
    "epochs": 150,
    "batch_size": 128,
    "learning_rate": 0.01,
    "lr_scheduler_type": 'linear',
    "lr_scheduler": True,
    "lr_scheduler_gamma": 0.95
}

real_dataset_list = ["mnist", 'letter', 'connect', 'census', 'forest', 'segment', 'satimages', 'pendigits', 'protein', 'sensit']

import os
import requests

def download(url: str, dst: str):    
    if os.path.exists(dst):
        print(f'Using dataset from: {dst}')
        return False

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    print(f'Downloading dataset from: {url}...', end='')
    r = requests.get(url)
    print('Done')
    print(f'Writing to: {dst}...', end='')
    if url.endswith('tgz'):
        os.mkdir(dst)
        tarobj = tarfile.open(fileobj=io.BytesIO(r.content))
        for i in tarobj.getnames():
            with open(f'{dst}/{os.path.basename(i)}', 'w') as f:
                f.write(tarobj.extractfile(i).read().decode('utf-8'))
        tarobj.close()

    elif url.endswith('zip'):
        os.mkdir(dst)
        import zipfile
        zipobj = zipfile.ZipFile(io.BytesIO(r.content))
        for i in zipobj.namelist():
            with zipobj.open(i) as zf:
                with open(f'{dst}/{i}', 'w') as f:
                    f.write(zf.read().decode('utf-8'))

        ## way2
        # with open('temp.zip', 'wb') as f:
        #     f.write(r.content)

        # with zipfile.ZipFile('temp.zip', 'r') as zip_ref:
        #     zip_ref.extractall(dst)
        # os.remove('temp.zip')            

    elif url.endswith('gz'):
        with open(dst, "wb") as file:
            file.write(r.content)

    else:
        if url.endswith('bz2'):
            towrite = bz2.decompress(r.content).decode('utf-8')
        else:
            towrite = r.text

        with open(dst, 'w') as f:
            f.write(towrite)

    print('Done')

    return True

def read_libsvm_format(
        file_path: str, n_features: int, n_classes: int, name: str='', shuffle_seed=1
    ):
        is_classification = (n_classes > 0)

        with open(file_path, 'r') as f:
            content = f.read()
        assert ':  ' not in content, 'Error while reading: {}'.format(file_path)

        content = content.replace(': ', ':')
        content = content.strip()
        lines = content.split('\n')
        lines = [line.strip() for line in lines]

        x = np.zeros((len(lines), n_features), dtype=np.float32)
        y = np.zeros((len(lines),), dtype=np.int64 if is_classification else np.float32)

        for line_idx, line in enumerate(lines):
            for unit_idx, unit in enumerate(line.split()):
                if unit_idx == 0:
                    assert ':' not in unit
                    if is_classification:
                        y[line_idx] = int(unit.strip())
                    else:
                        y[line_idx] = float(unit.strip())
                else:
                    feat, val = unit.strip().split(':')
                    feat: int = int(feat)
                    val: float = float(val)
                    x[line_idx][feat - 1] = val

        if is_classification:
            # To get classes in [0..n_classes-1]
            y = y - np.min(y)

        return x, y


def get_config(dataset_code):
    for config in [config_MN, config_LT, config_C4, config_CN, config_FC, config_SI, config_SG, config_PD, config_PR, config_IT]:

        if config["code"] == dataset_code:
            return config

    raise Exception(f"Invalid dataset code {dataset_code}.")


def load_dataset(config, standardize=True):
    if config["code"] == "mnist":
        # load dataset
        data_folder = config["filepath"]
        if not config["preprocess"]:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        kwargs = {'num_workers': 2, 'pin_memory': True}
        trainset = torchvision.datasets.MNIST(root=data_folder, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=config["batch_size"], shuffle=True, **kwargs)
        testset = torchvision.datasets.MNIST(root=data_folder, train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=config["batch_size"], shuffle=False, num_workers=2)
        return train_loader, None, test_loader

    elif config["code"] == "letter":
        def read_txt(cat="train"):
            file_path = config["filepath"] + "letter-"+ cat + ".txt"
            with open(file_path, 'r') as f:
                content = f.read()
            assert ':  ' not in content, 'Error while reading: {}'.format(file_path)

            content = content.replace(': ', ':')
            content = content.strip()
            lines = content.split('\n')
            lines = [line.strip() for line in lines]

            x = np.zeros((len(lines), config['n_attributes']), dtype=np.float32)
            y = np.zeros((len(lines),), dtype=np.int64)

            for line_idx, line in enumerate(lines):
                for unit_idx, unit in enumerate(line.split()):
                    if unit_idx == 0:
                        assert ':' not in unit
                        y[line_idx] = int(unit.strip())
                       
                    else:
                        feat, val = unit.strip().split(':')
                        feat: int = int(feat)
                        val: float = float(val)
                        x[line_idx][feat - 1] = val

            # print(np.min(y), np.max(y))
            y = y - np.min(y)
            
            return x, y
        
        X_train, y_train = read_txt("train")
        X_test, y_test = read_txt("test")
        X_val, y_val = read_txt("val")
        
        if config["preprocess"]:
            scaler = StandardScaler().fit(X_train)
            # from sklearn.preprocessing import RobustScaler
            # scaler = RobustScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
        
        # Create TensorDataset for X_train and y_train with batch dimension
        train_dataset = TensorDataset(torch.Tensor(X_train).to(torch.float32), torch.Tensor(y_train).to(torch.int64))

        # Create DataLoader for training set
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

        # Create TensorDataset for X_test and y_test with batch dimension
        test_dataset = TensorDataset(torch.Tensor(X_test).to(torch.float32), torch.Tensor(y_test).to(torch.int64))

        # Create DataLoader for testing set
        test_dataloader = DataLoader(test_dataset, batch_size=len(X_test), shuffle=False)

        # Create TensorDataset for X_val and y_val with batch dimension
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val).to(torch.int64))

        # Create DataLoader for validation set
        val_dataloader = DataLoader(val_dataset, batch_size=len(X_val), shuffle=False)
        return train_dataloader, val_dataloader, test_dataloader
    
    elif config["code"] == "connect":
        def read_txt(cat="connect-4"):
            file_path = config["filepath"] + cat + ".txt"
            with open(file_path, 'r') as f:
                content = f.read()
            assert ':  ' not in content, 'Error while reading: {}'.format(file_path)

            content = content.replace(': ', ':')
            content = content.strip()
            lines = content.split('\n')
            lines = [line.strip() for line in lines]

            x = np.zeros((len(lines), config['n_attributes']), dtype=np.float32)
            y = np.zeros((len(lines),), dtype=np.int64)

            for line_idx, line in enumerate(lines):
                for unit_idx, unit in enumerate(line.split()):
                    if unit_idx == 0:
                        assert ':' not in unit
                        y[line_idx] = int(unit.strip())
                       
                    else:
                        feat, val = unit.strip().split(':')
                        feat: int = int(feat)
                        val: float = float(val)
                        x[line_idx][feat - 1] = val

            # print(np.min(y), np.max(y))
            y = y - np.min(y)
            
            return x, y
        X, y = read_txt("connect-4")
        X, X_test, y, y_test = train_test_split(X, y, test_size=config["test_split"], random_state=config["tseed"])
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config["val_split"], random_state=config["tseed"])
        if config["preprocess"]:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
        
        # Create TensorDataset for X_train and y_train with batch dimension
        train_dataset = TensorDataset(torch.Tensor(X_train).to(torch.float32), torch.Tensor(y_train).to(torch.int64))
        # Create DataLoader for training set
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        # Create TensorDataset for X_test and y_test with batch dimension
        test_dataset = TensorDataset(torch.Tensor(X_test).to(torch.float32), torch.Tensor(y_test).to(torch.int64))
        # Create DataLoader for testing set
        test_dataloader = DataLoader(test_dataset, batch_size=len(X_test), shuffle=False)
        
        # Create TensorDataset for X_val and y_val with batch dimension
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val).to(torch.int64))
        # Create DataLoader for validation set
        val_dataloader = DataLoader(val_dataset, batch_size=len(X_val), shuffle=False)
        return train_dataloader, val_dataloader, test_dataloader
    
    elif config["code"] == "census":
        df = pd.read_csv(config["filepath"] + "USCensus1990.data.txt")
        split_at = int(len(df) * 0.8)  # Split data into 80% train, 10% validation, and 10% test
        
        train = df.iloc[:split_at].iloc[:, 1:]
        test = df.iloc[split_at:].iloc[:, 1:]

        y =  train['iYearsch'].to_numpy(np.int64)
        X = train.drop('iYearsch', axis=1).to_numpy(np.float32)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config["val_split"], random_state=config["tseed"])
        y_test = test['iYearsch'].to_numpy(np.int64)
        X_test = test.drop('iYearsch', axis=1).to_numpy(np.float32)

    
        if config["preprocess"]:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

        # Create TensorDataset for X_train and y_train with batch dimension
        train_dataset = TensorDataset(torch.Tensor(X_train).to(torch.float32), torch.Tensor(y_train).to(torch.int64))
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        test_dataset = TensorDataset(torch.Tensor(X_test).to(torch.float32), torch.Tensor(y_test).to(torch.int64))
        test_dataloader = DataLoader(test_dataset, batch_size=len(X_test), shuffle=False)

        # Create TensorDataset for X_val and y_val with batch dimension
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val).to(torch.int64))
        # Create DataLoader for validation set
        val_dataloader = DataLoader(val_dataset, batch_size=len(X_val), shuffle=False)
        return train_dataloader, val_dataloader, test_dataloader
    
    elif config["code"] == "forest":
        with gzip.open(config["filepath"] + "covtype.data.gz", 'rb') as f:
            data = pd.read_csv(f, header=None, names=config['attributes'], index_col=False)
        first_column = data.pop('Cover_Type')
        # insert column using insert(position,column_name,first_column) function
        data.insert(0, 'Cover_Type', first_column)
        data['Cover_Type'] = data['Cover_Type'].apply(lambda x: x - 1)
        
        # Split the dataset into 80% training and 20% testing
        data, test = train_test_split(data, test_size=config["test_split"], random_state=config["tseed"], stratify=data['Cover_Type'])
        train, val = train_test_split(data, test_size=config["val_split"], random_state=config["tseed"], stratify=data['Cover_Type'])
        
    
        X_cat_train = train.iloc[:, 11:55].values
        B = train.iloc[:, 55:60]
        A = train.iloc[:, 1:11]
        X_num_train = pd.concat([A, B], axis = 1).values
        y_train = train.iloc[:, 0].values

        X_cat_val = val.iloc[:, 11:55].values
        B = val.iloc[:, 55:60]
        A = val.iloc[:, 1:11]
        X_num_val = pd.concat([A, B], axis = 1).values
        y_val = val.iloc[:, 0].values

        X_cat_test = test.iloc[:, 11:55].values
        B = test.iloc[:, 55:60]
        A = test.iloc[:, 1:11]
        X_num_test = pd.concat([A, B], axis = 1).values
        y_test = test.iloc[:, 0].values
       
        if config["preprocess"]:
            scaler = StandardScaler()
            # fit to training data
            scaler.fit(X_num_train)
            X_num_train = scaler.transform(X_num_train)
            X_num_val = scaler.transform(X_num_val)
            X_num_test = scaler.transform(X_num_test)
        X_train = np.hstack((X_num_train, X_cat_train))
        X_val = np.hstack((X_num_val, X_cat_val))
        X_test = np.hstack((X_num_test, X_cat_test))

        # print(X_train.shape, X_val.shape, X_test.shape)
        # exit()
        
        # Create TensorDataset for X_train and y_train with batch dimension
        train_dataset = TensorDataset(torch.Tensor(X_train).to(torch.float32), torch.Tensor(y_train).to(torch.int64))
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        test_dataset = TensorDataset(torch.Tensor(X_test).to(torch.float32), torch.Tensor(y_test).to(torch.int64))
        test_dataloader = DataLoader(test_dataset, batch_size=len(X_test), shuffle=False)

        # Create TensorDataset for X_val and y_val with batch dimension
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val).to(torch.int64))
        # Create DataLoader for validation set
        val_dataloader = DataLoader(val_dataset, batch_size=len(X_val), shuffle=False)
        return train_dataloader, val_dataloader, test_dataloader
        
    elif config["code"] == "segment":
        url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/segment.scale'
        path = f'{config["filepath"]}segment.txt'
        download(url, path)

        X, y = read_libsvm_format(
            path,
            n_features=config['n_attributes'],
            n_classes=config['n_classes'],
            name='segment', shuffle_seed=config['tseed'])
        
        X, X_test, y, y_test = train_test_split(X, y, test_size=config["test_split"], random_state=config["tseed"])
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config["val_split"], random_state=config["tseed"])
        
        if config["preprocess"]:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
        
        # Create TensorDataset for X_train and y_train with batch dimension
        train_dataset = TensorDataset(torch.Tensor(X_train).to(torch.float32), torch.Tensor(y_train).to(torch.int64))
        # Create DataLoader for training set
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        
        # Create TensorDataset for X_test and y_test with batch dimension
        test_dataset = TensorDataset(torch.Tensor(X_test).to(torch.float32), torch.Tensor(y_test).to(torch.int64))
        # Create DataLoader for testing set
        test_dataloader = DataLoader(test_dataset, batch_size=len(X_test), shuffle=False)
        
        # Create TensorDataset for X_val and y_val with batch dimension
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val).to(torch.int64))
        # Create DataLoader for validation set
        val_dataloader = DataLoader(val_dataset, batch_size=len(X_val), shuffle=False)
        return train_dataloader, val_dataloader, test_dataloader
    
    elif config["code"] == "satimages":
        url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/satimage.scale.tr'
        path = f'{config["filepath"]}satimage-train.txt'
        download(url, path)
        X_train, y_train = read_libsvm_format(
            path,
            n_features=config['n_attributes'],
            n_classes=config['n_classes'],
            name='satimages', shuffle_seed=config['tseed'])

        url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/satimage.scale.t'
        path = f'{config["filepath"]}satimage-test.txt'
        download(url, path)
        X_test, y_test = read_libsvm_format(
            path,
            n_features=config['n_attributes'],
            n_classes=config['n_classes'],
            name='satimages', shuffle_seed=config['tseed'])
        
        url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/satimage.scale.val'
        path = f'{config["filepath"]}satimage-val.txt'
        download(url, path)
        X_val, y_val = read_libsvm_format(
            path,
            n_features=config['n_attributes'],
            n_classes=config['n_classes'],
            name='satimages', shuffle_seed=config['tseed'])
        
        
        if config["preprocess"]:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
        
        # Create TensorDataset for X_train and y_train with batch dimension
        train_dataset = TensorDataset(torch.Tensor(X_train).to(torch.float32), torch.Tensor(y_train).to(torch.int64))
        # Create DataLoader for training set
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        
        # Create TensorDataset for X_test and y_test with batch dimension
        test_dataset = TensorDataset(torch.Tensor(X_test).to(torch.float32), torch.Tensor(y_test).to(torch.int64))
        # Create DataLoader for testing set
        test_dataloader = DataLoader(test_dataset, batch_size=len(X_test), shuffle=False)
        
        # Create TensorDataset for X_val and y_val with batch dimension
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val).to(torch.int64))
        # Create DataLoader for validation set
        val_dataloader = DataLoader(val_dataset, batch_size=len(X_val), shuffle=False)
        return train_dataloader, val_dataloader, test_dataloader
    
    elif config["code"] == "pendigits":
        url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits'
        path = f'{config["filepath"]}pendigits.txt'
        download(url, path)
        X, y = read_libsvm_format(
            path,
            n_features=config['n_attributes'],
            n_classes=config['n_classes'],
            name='pendigits', shuffle_seed=config['tseed'])

        url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits.t'
        path = f'{config["filepath"]}pendigits-test.txt'
        download(url, path)
        X_test, y_test = read_libsvm_format(
            path,
            n_features=config['n_attributes'],
            n_classes=config['n_classes'],
            name='pendigits', shuffle_seed=config['tseed'])
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config["val_split"], random_state=config["tseed"])
        
        if config["preprocess"]:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
        
        # Create TensorDataset for X_train and y_train with batch dimension
        train_dataset = TensorDataset(torch.Tensor(X_train).to(torch.float32), torch.Tensor(y_train).to(torch.int64))
        # Create DataLoader for training set
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        
        # Create TensorDataset for X_test and y_test with batch dimension
        test_dataset = TensorDataset(torch.Tensor(X_test).to(torch.float32), torch.Tensor(y_test).to(torch.int64))
        # Create DataLoader for testing set
        test_dataloader = DataLoader(test_dataset, batch_size=len(X_test), shuffle=False)
        
        # Create TensorDataset for X_val and y_val with batch dimension
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val).to(torch.int64))
        # Create DataLoader for validation set
        val_dataloader = DataLoader(val_dataset, batch_size=len(X_val), shuffle=False)
        return train_dataloader, val_dataloader, test_dataloader
    
    elif config["code"] == "protein":
        url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/protein.tr.bz2'
        path = f'{config["filepath"]}protein-train.txt'
        download(url, path)
        X_train, y_train = read_libsvm_format(
            path,
            n_features=config['n_attributes'],
            n_classes=config['n_classes'],
            name='satimages', shuffle_seed=config['tseed'])

        url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/protein.t.bz2'
        path = f'{config["filepath"]}protein-test.txt'
        download(url, path)
        X_test, y_test = read_libsvm_format(
            path,
            n_features=config['n_attributes'],
            n_classes=config['n_classes'],
            name='satimages', shuffle_seed=config['tseed'])
        
        url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/protein.val.bz2'
        path = f'{config["filepath"]}protein-val.txt'
        download(url, path)
        X_val, y_val = read_libsvm_format(
            path,
            n_features=config['n_attributes'],
            n_classes=config['n_classes'],
            name='satimages', shuffle_seed=config['tseed'])
        
        if config["preprocess"]:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
        
        # Create TensorDataset for X_train and y_train with batch dimension
        train_dataset = TensorDataset(torch.Tensor(X_train).to(torch.float32), torch.Tensor(y_train).to(torch.int64))
        # Create DataLoader for training set
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        
        # Create TensorDataset for X_test and y_test with batch dimension
        test_dataset = TensorDataset(torch.Tensor(X_test).to(torch.float32), torch.Tensor(y_test).to(torch.int64))
        # Create DataLoader for testing set
        test_dataloader = DataLoader(test_dataset, batch_size=len(X_test), shuffle=False)
        
        # Create TensorDataset for X_val and y_val with batch dimension
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val).to(torch.int64))
        # Create DataLoader for validation set
        val_dataloader = DataLoader(val_dataset, batch_size=len(X_val), shuffle=False)
        return train_dataloader, val_dataloader, test_dataloader
    
    elif config["code"] == "sensit":
        url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/combined.bz2'
        path = f'{config["filepath"]}sensit-combined.txt'
        download(url, path)
        X, y = read_libsvm_format(
            path,
            n_features=config['n_attributes'],
            n_classes=config['n_classes'],
            name='sensit', shuffle_seed=config['tseed'])

        url = r'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/combined.t.bz2'
        path = f'{config["filepath"]}sensit-combined-test.txt'
        download(url, path)
        X_test, y_test = read_libsvm_format(
            path,
            n_features=config['n_attributes'],
            n_classes=config['n_classes'],
            name='sensit', shuffle_seed=config['tseed'])
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config["val_split"], random_state=config["tseed"])
        
        
        if config["preprocess"]:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
        
        # Create TensorDataset for X_train and y_train with batch dimension
        train_dataset = TensorDataset(torch.Tensor(X_train).to(torch.float32), torch.Tensor(y_train).to(torch.int64))
        # Create DataLoader for training set
        train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        
        # Create TensorDataset for X_test and y_test with batch dimension
        test_dataset = TensorDataset(torch.Tensor(X_test).to(torch.float32), torch.Tensor(y_test).to(torch.int64))
        # Create DataLoader for testing set
        test_dataloader = DataLoader(test_dataset, batch_size=len(X_test), shuffle=False)
        
        # Create TensorDataset for X_val and y_val with batch dimension
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val).to(torch.int64))
        # Create DataLoader for validation set
        val_dataloader = DataLoader(val_dataset, batch_size=len(X_val), shuffle=False)
        return train_dataloader, val_dataloader, test_dataloader
    



if __name__ == "__main__":
    pdb.set_trace()