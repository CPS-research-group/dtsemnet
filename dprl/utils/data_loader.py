
import sys
sys.path.append(".")

import csv
import pdb
import os

import numpy as np
import pandas as pd
import gzip
import pickle
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file

import openml
#openml.config.apikey = 'FILL_IN_OPENML_API_KEY'  # set the OpenML Api Key
## Use following to get the task id of the benchmark suite, it's already listed
# SUITE_ID = 336 # Regression on numerical features
# benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite

###### list of datasets
dataset_list = ["abalone",
                "ailerons",
                "cpu_active",
                "pdb_bind",
                "year",
                "ctslice",
                "ms",
                
                "wine_quality",
                "superconduct",
                "pol",
                "elevators",
                "Bike_Sharing_Demand",
                "sulfur",
                "medical",

                "houses",
                "taxi",

                "synth_30k",
                "synth_5k",
                "synth_5l"
                ]

#
#"houses",
#"house_16H",
# "taxi", 

######

### Configs for OpenML datasets
## Paper: Why do tree-based models still outperform deep learning on tabular data?
# https://github.com/LeoGrin/tabular-benchmark?tab=readme-ov-file
## Already Done
# dataset: cpu_act 361072
# dataset: abalone 361280
# dataset: Ailerons 361077
# dataset: yprop_4_1 361279 (year)

## New
### first set
# dataset: wine_quality 361076
# dataset: superconduct 361088
# dataset: pol 361073
# dataset: elevators 361074
# dataset: Bike_Sharing_Demand 361082
# dataset: sulfur 361085
# dataset: houses 361078

### second set
# dataset: house_16H 361079
# dataset: house_sales 361084
# dataset: MiamiHousing2016 361087
# dataset: Brazilian_houses 361081

# dataset: delays_zurich_transport 361281

### third large
# dataset: medical 361086
# dataset: nyc-taxi-green-dec-2016 361083

# dataset: diamonds 361080


def load_pklgz(filename):
    print("... loading", filename)
    with gzip.open(filename, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def load_dataset(config, standardize=True, seed=1):
    if config["code"] == "abalone":
        npz = np.load(f"datasets/abalone/abalone{config['tseed']-1}.npz")
        X = npz["X_train"]
        y = npz["y_train"]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config["val_split"], random_state=1)
        X_test = npz["X_test"]
        y_test = npz["y_test"]

    elif config["code"] == "ailerons":
        #==== Convert categorical to one hot
        ailerons_data = pd.read_csv(f"{config['filepath']}ailerons.data", names=config["attributes"])
        ailerons_test = pd.read_csv(f"{config['filepath']}ailerons.test", names=config["attributes"])
        
        X = np.array(ailerons_data.iloc[:, :-1], dtype=np.float32)
        y = np.array(ailerons_data.iloc[:, -1], dtype=np.float32) * 1e4 #info: scale y to be in the same range as x

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config["val_split"], random_state=1)

        X_test = np.array(ailerons_test.iloc[:, :-1], dtype=np.float32)
        y_test = np.array(ailerons_test.iloc[:, -1], dtype=np.float32) * 1e4

    
    elif config["code"] == "cpu_active":
        cactv_data = pd.read_csv(f"{config['filepath']}comp.data", header=None, sep=' ')
        
        X = cactv_data.iloc[:, 1:22].values
        y = cactv_data.iloc[:, 23].values

        X = X.astype(float)
        y = y.astype(float)

        X, X_test, y, y_test = train_test_split(X, y, test_size=config["test_split"], random_state=config["tseed"])    
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config["val_split"], random_state=1)
        

    elif config["code"] == "pdb_bind":
        X_train, y_train, X_val, y_val, X_test, y_test = load_pklgz(f"{config['filepath']}PDBbind.pkl.gz")
        
    elif config["code"] == "year":
        X = np.load(f"{config['filepath']}year-train-x.npy").astype(np.float32)
        y = np.load(f"{config['filepath']}year-train-y.npy").astype(np.float32)
        X_test = np.load(f"{config['filepath']}year-test-x.npy").astype(np.float32)
        y_test = np.load(f"{config['filepath']}year-test-y.npy").astype(np.float32)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config["val_split"], random_state=1)
    
    elif config["code"] == "ctslice":
        df = pd.read_csv(f'{config["filepath"]}/slice_localization_data.csv')
        df=df.drop("patientId",axis=1)
        y=df["reference"].values
        X=df.drop("reference",axis=1).values
        
        # First, split into train + val and test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=config["test_split"],
            random_state=config["tseed"],
        )

        # Then, split train + val into train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=config["val_split"],
            random_state=1,
        )
        
    elif config["code"] == "ms":
        if not os.path.exists(f"{config['filepath']}ms.npz"):
            X_train, y_train, _ = load_svmlight_file(config['filepath'] + "train.txt" , dtype=np.float32, query_id=True)
            X_train = X_train.toarray()

            X_vali, y_vali, _ = load_svmlight_file(config['filepath'] + "vali.txt" , dtype=np.float32, query_id=True)
            X_vali = X_vali.toarray()

            X_test, y_test, _ = load_svmlight_file(config['filepath'] + "test.txt" , dtype=np.float32, query_id=True)
            X_test = X_test.toarray()
            
            np.savez_compressed(f"{config['filepath']}ms.npz",
                        X_train=X_train, y_train=y_train,
                        X_vali=X_vali, y_vali=y_vali,
                        X_test=X_test, y_test=y_test)
        
        data = np.load(f"{config['filepath']}ms.npz")
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_vali'], data['y_vali']
        X_test, y_test = data['X_test'], data['y_test']
    
    elif config["code"] == "medical":
        task = openml.tasks.get_task(361086)  # download the OpenML task
        dataset = task.get_dataset()
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        X = X.values
        y = y.values
        y = y * 1e1  # scale y to be in the same range as x
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.7, random_state=1)
        
    elif config["code"] == "wine_quality":
        task = openml.tasks.get_task(361076)  # download the OpenML task
        dataset = task.get_dataset()
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        X = X.values
        y = y.values
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.7, random_state=1)
        

    elif config["code"] == "superconduct":
        task = openml.tasks.get_task(361088)  # download the OpenML task
        dataset = task.get_dataset()
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        X = X.values
        y = y.values
        y = y * 1e-1  # scale y to be in the same range as x
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.7, random_state=1)
        



    elif config["code"] == "pol":
        task = openml.tasks.get_task(361073)  # download the OpenML task
        dataset = task.get_dataset()
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        X = X.values
        y = y.values
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.7, random_state=1)
        

   

    elif config["code"] == "elevators":
        task = openml.tasks.get_task(361074)
        dataset = task.get_dataset()
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        X = X.values
        y = y.values
        y = y * 1e2  # scale y
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.7, random_state=1)
        print("Loaded dataset: elevators")

    elif config["code"] == "Bike_Sharing_Demand":
        task = openml.tasks.get_task(361082)
        dataset = task.get_dataset()
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        X = X.values
        y = y.values
        y = y * 1e-2  # scale y
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.7, random_state=1)
        print("Loaded dataset: Bike_Sharing_Demand")

    elif config["code"] == "taxi":
        task = openml.tasks.get_task(361083)
        dataset = task.get_dataset()
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        X = X.values
        y = y.values
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.7, random_state=1)
        print("Loaded dataset: nyc-taxi-green-dec-2016")

    elif config["code"] == "sulfur":
        task = openml.tasks.get_task(361085)
        dataset = task.get_dataset()
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        X = X.values
        y = y.values
        y = y * 1e1  # scale y
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.7, random_state=1)
        print("Loaded dataset: sulfur")
    
    elif config["code"] == "houses":
        
        task = openml.tasks.get_task(361078)
        dataset = task.get_dataset()
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        X = X.values
        y = y.values

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.7, random_state=1)
        print("Loaded dataset: houses")

    elif config["code"] == "synth_30k":
        # Load the synthetic dataset
        # data = np.load('datasets/synthetic/synth_data_30k.npz')
        data = np.load(config['filepath'] + 'synth_data_30k.npz')
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_test'] # Note: In the synthetic dataset, 'X_test' is used for validation
        y_val = data['y_test']
        X_test = data['X_test']
        y_test = data['y_test']
        print("Loaded dataset: Synth_30k")
    elif config["code"] == "synth_5k":
        # Load the synthetic dataset
        data = np.load('datasets/synthetic/synth_data_5k.npz')
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_test'] # Note: In the synthetic dataset, 'X_test' is used for validation
        y_val = data['y_test']
        X_test = data['X_test']
        y_test = data['y_test']
        print("Loaded dataset: Synth_5k")
    elif config["code"] == "synth_5l":
        # Load the synthetic dataset
        data = np.load('datasets/synthetic/synth_data_5l.npz')
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_test'] # Note: In the synthetic dataset, 'X_test' is used for validation
        y_val = data['y_test']
        X_test = data['X_test']
        y_test = data['y_test']
        print("Loaded dataset: Synth_5l")
      

    else:
        raise ValueError(f"Dataset {config['code']} not supported.")
    
    # print(f"X size: {X.shape}, y size: {y.shape}")
    print(f"X_train size: {X_train.shape}, y_train size: {y_train.shape}")
    print(f"X_val size: {X_val.shape}, y_val size: {y_val.shape}")
    print(f"X_test size: {X_test.shape}, y_test size: {y_test.shape}")
    # exit()
    ##############################
    scaler_y = None
    if standardize:
        from sklearn.preprocessing import MaxAbsScaler, QuantileTransformer, StandardScaler
        
        #### Scale X values
        ###################
        # scaler_x = QuantileTransformer(output_distribution="normal").fit(X_train)
        # scaler_x = QuantileTransformer(output_distribution="uniform").fit(X_train)
        scaler_x = StandardScaler().fit(X_train)
        X_train = scaler_x.transform(X_train)
        X_val = scaler_x.transform(X_val)
        X_test = scaler_x.transform(X_test)


        #### Scale y values
        ###################
        #MinMaxScaler()
        scaler_y = StandardScaler().fit(y_train.reshape(-1,1))
        y_train = scaler_y.transform(y_train.reshape(-1,1)).reshape(-1)
        y_val = scaler_y.transform(y_val.reshape(-1,1)).reshape(-1) 
        y_test = scaler_y.transform(y_test.reshape(-1,1)).reshape(-1)
    
    # import matplotlib.pyplot as plt
    # plt.hist(X_train.flatten(), bins=100)
    # plt.show()
    # # save the plot
    # plt.savefig("X_train.png")
    # exit()
    # print max and min of X_train
    print("X_train min:", X_train.min())
    print("X_train max:", X_train.max())
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler_y

