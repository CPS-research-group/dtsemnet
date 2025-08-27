import pdb
import math
import argparse
import copy
import sys
from datetime import datetime
import json
import pickle
import os

import time
import numpy as np

from itertools import chain
from rich import print


import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
torch.backends.cudnn.deterministic = True
## DTSemNet
from dprl.agents.dtsemnet_ste import DTSemNet
from dprl.agents.dgt import DGT
##

from dprl.utils.data_loader import load_dataset, dataset_list
from dprl.utils.config_loader import get_config

# Import utility functions
from dprl.utils.logs import save_histories_to_file
from dprl.utils.eval_agent import evaluate, get_leaf_distribution

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DT*-Net training script.')
    parser.add_argument('-i', '--dataset', help="What dataset to use?", required=True, type=str)
    parser.add_argument('-m', '--model', help="Which model to use?", required=True, type=str)
    parser.add_argument('-s', '--simulations', help="How many simulations?", required=False, type=int, default=1)
    parser.add_argument('--output_prefix', help='Which output name to use?', required=False, default="dtsemnet", type=str)
    parser.add_argument('--should_normalize_dataset', help='Should normalize dataset?', required=False, default=True,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=True,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('-g', '--gpu', help='gpu?', required=False, default=False, action='store_true')
    
    args = vars(parser.parse_args())

    # Initialization
    simulations = args["simulations"]
    mname = args['model'] # model name
    use_gpu = args['gpu']
    

    command_line = str(args)
    command_line += "\n\npython -m reg_dataset_linear.reg_train_linear " + " ".join(
        [f"--{key} {val}" for (key, val) in args.items()]) + "\n\n---\n\n"
    curr_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    # todo: change output file name
    output_path_summ = f"logs/regression/{mname}_{args['output_prefix']}_{curr_time}_summary.txt"
    output_path_full = f"logs/regression/{mname}_{args['output_prefix']}_{curr_time}_full.txt"

    
    
    
    # 1. Load Default Configuration
    if args['dataset'].endswith('all'):
        data_configs = [get_config(d, mname ,"regression") for d in dataset_list]
    elif args['dataset'].endswith("onwards"):
        dataset_start = dataset_list.index(args['dataset'][:-len("_onwards")])
        data_configs = [get_config(d, mname ,"regression") for d in dataset_list[dataset_start:]]
    else:
        data_configs = [get_config(args['dataset'], mname ,"regression")]

    # info:
        # data_configs: list of dictionaries, each dictionary contains information about dataset
    


    histories = []
    hconfig = []
    
    
    for dataset_id, data_config in enumerate(data_configs):
       print(f'===>Training Started for Configuration:  Depth:{data_config["depth"]}<===')
       print(data_config)
       
       for seed in range(data_config["nseed"]): 
            data_config["tseed"] = seed+1
            hconfig.append(data_config.copy())
            
            ## 2. Load Data
            X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_dataset(data_config, standardize=args["should_normalize_dataset"]) 
            
            N = X_train.shape[0]
            print('train size:', X_train.shape, 'test size:', X_test.shape, 'val size:', X_val.shape)
            
            # info: list of histories for each dataset [[h_ds1], [h_ds2], []]
            histories.append([]) 

            # info: Loop over each simulation
            # need to implement for 100 simulations
            for simulation in range(simulations):
                # info: seed for torch, random and numpy for producable results
                torch.manual_seed(simulation)
                np.random.seed(simulation)
                torch.cuda.manual_seed(simulation)
                
                ### TRAIN ###
                print("=" * 70)
                print(
                    f"[green]Iteration {simulation}/{args['simulations']}[/green] [blue] Seed {seed+1} [/blue] [red](dataset: {data_config['name']}, {dataset_id}/{len(data_configs)}):[/red] [yellow] Num Sample: {N} [/yellow]")
                print("=" * 70)

                ########################
                ########################
                
            
                start_time = time.perf_counter()
                tr_loss = []
                te_loss = []
                tr_acc = []


                ######### 3. Training
                # FULL TRAINING LOOP


                # 1. Inverse scale targets
                y_train_true = scaler.inverse_transform(y_train.reshape(-1, 1)).reshape(-1)
                y_test_true = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

                # 2. Define base model
                model = DecisionTreeRegressor(random_state=simulation)

                # 3. Define hyperparameter grid
                param_grid = {
                    'max_depth': [None],
                    'min_samples_leaf': [10, 20],
                    'min_samples_split': [20, 50],
                    'ccp_alpha': [0.01, 0.05, 0.1],
                    'criterion': ['squared_error'],
                    'splitter': ['random']
                }

                # 4. Set up cross-validation
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=5,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=1
                )

                # 5. Fit the best model
                grid_search.fit(X_train, y_train_true)

                # 6. Best estimator
                model = grid_search.best_estimator_

                print(f"Best parameters found: {grid_search.best_params_}")
                #print depth of the final tree
                print(f"Final tree depth: {model.get_depth()}")

                # 7. Evaluate
                y_train_pred = model.predict(X_train)
                # y_train_true = scaler.inverse_transform(y_train.reshape(-1, 1)).reshape(-1)
                # y_train_pred = scaler.inverse_transform(y_train_pred.reshape(-1, 1)).reshape(-1)
                train_rmse = mean_squared_error(y_train_true, y_train_pred, squared=False)
                
                y_test_pred = model.predict(X_test)
                # y_test_pred = scaler.inverse_transform(y_test_pred.reshape(-1, 1)).reshape(-1)
                test_rmse = mean_squared_error(y_test_true, y_test_pred, squared=False)

                from sklearn.tree import export_text

                # Print tree as text
                tree_rules = export_text(model, feature_names=list(np.arange(X_train.shape[1])))
                print(tree_rules)
                
                ########## END
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                
                
                print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
                print(f"Elapsed time: {elapsed_time:.2f} seconds")
                
               
                
                # info: save results to file
                histories[-1].append((elapsed_time,
                                    (N),
                                    (train_rmse, test_rmse),
                                    (0, 0),
                                    (0, 0),
                                    ))
                save_histories_to_file(hconfig, histories, output_path_summ, output_path_full, command_line)

                if args["verbose"]:
                    print(f"Saved summary to '{output_path_summ}'.")
                    print(f"Saved full data to '{output_path_full}'.")
    
    