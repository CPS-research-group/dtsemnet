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
from pprint import pformat 


import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
torch.backends.cudnn.deterministic = True
## DTSemNet
from dprl.agents.dtsemnet_topk import DTSemNet
from dprl.agents.dgt import DGT
##

from dprl.utils.data_loader import load_dataset, dataset_list
from dprl.utils.config_loader import get_config

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import ElasticNetCV, RidgeCV

import matplotlib.pyplot as plt

# Import utility functions
from dprl.utils.logs import save_histories_to_file
from dprl.utils.eval_agent import evaluate, get_leaf_distribution


# MLP Model Definition
class RegNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        hidden = 128
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        # Orthogonal initialization
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)

    def forward(self, x, mode='train'):
        return self.layers(x), None
    
    def update_temperature(self, epoch, max_epoch):
        pass




def fine_tune_regressor(X_train, y_train, model, loss, scaler, use_gpu, tr_loss, te_loss, tr_acc, mix_experts):
    print("Training Stage 2: Regression Layer")
    
    
    if use_gpu:
        test_op,_ = model(X_train.cuda(), mode='test')
    else:
        test_op,_ = model(X_train, mode='test')
    # selected_expert_op = model.selected_experts.detach().cpu().numpy()
    selected_expert_op = model.selected_experts.detach().cpu()
    topk_vals, topk_inds = torch.topk(selected_expert_op, 4, dim=1)
    # print(topk_vals)
    # print(topk_inds)
    selected_expert_top1 = topk_inds[:, 0].numpy()  # Top-1 expert index
    selected_expert_top2 = topk_inds[:, 1].numpy()  # Top-2 expert index
    selected_expert_top3 = topk_inds[:, 2].numpy()  # Top-3 expert index
    selected_expert_top4 = topk_inds[:, 3].numpy()
    
    # print(selected_expert_top1)
    # print('top2')
    # print(selected_expert_top2)
    # exit()
    # selected_expert_top1 = np.argmax(selected_expert_op, axis=1)
    
    

    #####
    ##### Plotting
    # # Count the occurrences of each expert
    expert_counts = np.bincount(selected_expert_top1, minlength=model.selected_experts.shape[1])

    # Increase figure size to reduce overlap
    plt.figure(figsize=(6, 4))  

    # Plot the histogram
    plt.bar(range(len(expert_counts)), expert_counts)
    plt.xlabel('Expert Index')
    plt.ylabel('Count')
    plt.title(f'CTSlice Top-8 (then argmax) before fine-tuning regression ') 

    # Set x-axis tick labels at an interval of 1 and rotate for better readability
    plt.xticks(range(len(expert_counts)), rotation=45, ha="right")  

    plt.tight_layout()  # Adjust layout to fit labels properly
    plt.savefig('expert_counts.png', dpi=300, bbox_inches='tight')
    
    # exit()
    want_plot = False
    if want_plot:
        num_experts = model.selected_experts.shape[1]

        # Count occurrences for each top-k rank
        count_top1 = np.bincount(selected_expert_top1, minlength=num_experts)
        count_top2 = np.bincount(selected_expert_top2, minlength=num_experts)
        count_top3 = np.bincount(selected_expert_top3, minlength=num_experts)

        expert_indices = np.arange(num_experts)

        # Plot stacked bar chart
        plt.figure(figsize=(8, 5))

        bar1 = plt.bar(expert_indices, count_top1, label='Top-1', color='#1f77b4')
        bar2 = plt.bar(expert_indices, count_top2, bottom=count_top1, label='Top-2', color='#ff7f0e')
        # bar3 = plt.bar(expert_indices, count_top3, bottom=count_top1 + count_top2, label='Top-3', color='#2ca02c')

        plt.xlabel('Expert Index')
        plt.ylabel('Number of Samples')
        plt.title('Expert Selections: First, Second')

        plt.xticks(expert_indices, rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        plt.savefig('expert_counts_stacked.png', dpi=300, bbox_inches='tight')

    # exit()  
    
    # extract data with class label 0
    for i in range(n_leaves):
        print('===========', i, '===========', i)
        
        
        # indices = selected_expert_top1== i
        if mix_experts == 1:
            indices = (selected_expert_top1 == i)
        elif mix_experts == 2:
            indices = (selected_expert_top1 == i) | (selected_expert_top2 == i)
        elif mix_experts == 3:
            indices =(selected_expert_top1 == i) | (selected_expert_top2 == i) | (selected_expert_top3 == i)
        elif mix_experts == 4:
            indices =(selected_expert_top1 == i) | (selected_expert_top2 == i) | (selected_expert_top3 == i) | (selected_expert_top4 == i)
        else:
            raise ValueError("num_experts should be 1, 2, 3 or 4")
        # 4 expers give 1.81
        # 3 experts give 1.65
        # 2 expeerts give 1.52
        ####### Select indices of SAMPLES

        # Step 1: Get boolean masks for each top-k level
        # top1_mask = (selected_expert_top1 == i)
        # top2_mask = (selected_expert_top2 == i)
        # top3_mask = (selected_expert_top3 == i)
        

        # # Step 2: Get the actual indices
        # top1_indices = np.where(top1_mask)[0]
        # top2_indices = np.where(top2_mask)[0]
        # top3_indices = np.where(top3_mask)[0]
        

        # # Step 3: Determine the number of samples to keep from top2 and top3
        # clip_size = len(top1_indices) // 1

        # # Step 4: Randomly sample (or truncate) top2 and top3 indices
        # # (shuffling for randomness)
        # np.random.shuffle(top2_indices)
        # np.random.shuffle(top3_indices)

        # top2_indices_clipped = top2_indices[:clip_size]
        # top3_indices_clipped = top3_indices[:clip_size]

        # # Optionally: do the same for top4 if needed
        # # top4_indices_clipped = top4_indices[:clip_size]  # if including top4 similarly

        # # Step 5: Combine all selected indices
        # indices = np.concatenate([top1_indices, top2_indices_clipped, top3_indices_clipped])

        ##########
        ########## Indices selected
        # If using for indexing tensors/arrays
        # selected_samples = X[final_indices]


        # Apply the filter to the tensors
        X_train_filtered = X_train[indices]
        y_train_filtered = y_train[indices]
        print(f"Leaf {i}: {len(X_train_filtered)} samples")

        if X_train_filtered.shape[0] == 0:
            continue
        
        ##### with Closed Form
        ######################
        # reg_model = LinearRegression()
        # reg_model = Ridge(alpha=15)
        # reg_model = Lasso(alpha=0.01)
        # reg_model = ElasticNet(alpha=10, l1_ratio=0.01)

        ### For top8
        if len(X_train_filtered)>4:
            reg_model = ElasticNetCV(
                        alphas=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5, 10.0, 12.0, 15.0, 20.0],       # cover both light and strong regularization
                        l1_ratio=[0.1, 0.5, 0.9, 1.0],              # blend from mostly L2 to mostly L1
                        cv=5,
                        random_state=42
                        )
        else:
            print('only <4 samples')
            # reg_model = Ridge(alpha=10)
            reg_model = Ridge(alpha=10)
            
            
        # reg_model = ElasticNetCV(
        #             alphas=[0.001, 0.01, 0.1, 1.0, 10.0],       # cover both light and strong regularization
        #             l1_ratio=[0.1, 0.5, 0.9, 1.0],              # blend from mostly L2 to mostly L1
        #             random_state=42
        #             )
        
        
        # reg_model = ElasticNetCV(
        #     alphas=np.logspace(-4, 2, 20),             # 10 values from 1e-4 to 1e2 (log-spaced)
        #     l1_ratio=np.linspace(0.1, 5.0, 20),        # 10 values from mostly L2 to pure L1
        #     cv=5,
        #     random_state=42,
        #     max_iter=10000                             # optional: ensure convergence on bigger data
        # )
        
        # reg_model = ElasticNetCV(
        #         alphas=[5, 8, 10, 12, 15],              # same alpha range as RidgeCV
        #         l1_ratio=[0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 2, 5], # test from mostly Ridge to Lasso
        #         cv=5,
        #         random_state=42,                          # just in case of slow convergence
        #     )

        # reg_model = RidgeCV(alphas=[0.001, 0.01, 0.1, 0.5, 1, 5, 8, 9, 10, 11, 12, 15, 20, 30], cv=5)
        
        reg_model.fit(X_train_filtered, y_train_filtered)
        weights = reg_model.coef_
        bias = reg_model.intercept_

        # print(weights, bias)
        # print(model.experts[i].layer)
        if use_gpu:
            model.experts[i].layer.weight.data = torch.tensor(weights, dtype=torch.float32).unsqueeze(0).cuda()
            model.experts[i].layer.bias.data = torch.tensor(bias, dtype=torch.float32).unsqueeze(0).cuda()
        else:
            model.experts[i].layer.weight.data = torch.tensor(weights, dtype=torch.float32).unsqueeze(0)
            model.experts[i].layer.bias.data = torch.tensor(bias, dtype=torch.float32).unsqueeze(0)

        if args["verbose"]:
            train_loss, multiv_acc_in = evaluate(model, train_dataloader, loss, scaler, use_gpu, 'test')
            test_loss, multiv_acc_test = evaluate(model, test_dataloader, loss, scaler, use_gpu, 'test')
            print(f"Stage2 | Leaf: {i}/{n_leaves}, Average Loss: {train_loss:.4f}, Train Acc: {multiv_acc_in:.4f} Test Acc: {multiv_acc_test:.4f}")
            tr_loss.append(train_loss)
            te_loss.append(test_loss)
            tr_acc.append(multiv_acc_in)
    # exit()




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
    
    tunning = False
    hist_leaf_dist = True
    
    # write hyperparameter search loop
    if tunning:
        temp_config = data_configs.pop()
        # repeate the dictionary in data_configs to   
        # ["hard_argmax_ste", "gumbel_softmax_ste", "entmax_ste",]  
        for use_L1 in [False, True]:
            for depth in [5, 6, 7, 8]:
                for batch_size in [32, 64, 128]:
                    for learning_rate in [0.0001, 0.001, 0.01, 0.1]:
                        for lr_scheduler_rate in [0.90, 0.95]:
                            for lambda_L1 in [1e-3, 1e-4, 1e-5]:
                                for topk in [2,4]:
                                    for mix_exp in [1,2]:
                                        temp_config["lr_scheduler_gamma"] = lr_scheduler_rate
                                        temp_config["learning_rate"] = learning_rate
                                        temp_config["batch_size"] = batch_size
                                        temp_config["lamda_L1"] = lambda_L1
                                        temp_config["depth"] = depth
                                        temp_config["use_L1"] = use_L1
                                        temp_config["top_k"] = topk
                                        temp_config["mix_experts"] = mix_exp
                                        
                                        data_configs.append(copy.deepcopy(temp_config))
        
        best_config = None
        best_acc = 100
    
    for dataset_id, data_config in enumerate(data_configs):
       print(f'===>Training Started for Configuration: Batch Size: {data_config["batch_size"]}, Learning Rate: {data_config["learning_rate"]}, LR Scheduler Rate: {data_config["lr_scheduler_gamma"]}<===')
       print(data_config)

       #Append the configuration(s) used
       command_line += "--- CONFIGURATION ---\n"
       command_line += pformat(data_configs)
       command_line += "\n--- END CONFIG ---\n\n"
       
       for seed in range(data_config["nseed"]): 
            data_config["tseed"] = seed+1
            hconfig.append(data_config.copy())
            
            ## 2. Load Data
            X_train, y_train, X_val, y_val, X_test, y_test, scaler = load_dataset(data_config, standardize=args["should_normalize_dataset"]) 
            # Prepare data as tensors    
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)
            print('train size:', X_train.shape, 'test size:', X_test.shape, 'val size:', X_val.shape)
            N = len(X_train)

            n_attributes = data_config["n_attributes"]
            n_classes = data_config["n_classes"]
            depth = data_config["depth"]
            n_inners, n_leaves = 2 ** depth - 1, 2 ** depth
            
            # if data_config["code"] == "abalone":
            #     bins = data_config["bins"][seed]
            #     # bins = np.linspace(-2.82, 0.35, 2)
            # else:
            bins = data_config["bins"]
            n_reg_classes = len(bins)
            print(bins)
              
            
            
            bin_indices = np.digitize(y_train, bins) - 1
            y_train_clf = torch.tensor(bin_indices, dtype=torch.int64)
            print(f"Max y_train_clf: {torch.max(y_train_clf).item()}")
            print(f"Min y_train_clf: {torch.min(y_train_clf).item()}")

            #==================
            unique_classes, class_counts = np.unique(y_train_clf, return_counts=True)

            bin_indices = np.digitize(y_test, bins) - 1
            y_test_clf = torch.tensor(bin_indices, dtype=torch.int64)
            print('train size:', X_train.shape, 'test size:', X_test.shape, 'val size:', X_val.shape)
            N = len(X_train)

            # Print the distribution of class labels
            print("Class Label Distribution:")
            for label, count in zip(unique_classes, class_counts):
                print(f"Class {label}: {count} samples")

            

            # Hyperparameters
            learning_rate = data_config["learning_rate"]
            batch_size = data_config["batch_size"]
            epochs = data_config["epochs"]
            
            
            # info: list of histories for each dataset [[h_ds1], [h_ds2], []]
            histories.append([]) 

            # dtsement does not support more classes than leaves
            if mname == 'dtsemnet_topk' and n_classes > n_leaves:
                print("=" * 70)
                print(f'Dataset {data_config["name"]}')
                print(f"[red]Error: Number of classes ({n_classes}) is greater than number of leaves ({n_leaves}).[/red]")
                print("=" * 70)
                # info: save results to file
                histories[-1].append((0,
                                    (N), \
                                    (0, 0)
                                    ))
                save_histories_to_file(data_configs, histories, output_path_summ, output_path_full, command_line)
                continue

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
                # Write Training Loop here
                # Define model and loss function
                # n_attributes = input feature
                # n_classes = output feature
                # depth = depth of tree
                # model = YourModel(n_attributes, n_classes)
                if mname == "mlp":
                    model = RegNN(n_attributes)
                elif mname == "dtregnet_topk":
                    model = DTSemNet(
                                    in_dim=n_attributes,
                                    out_dim=n_classes,
                                    height=depth,
                                    is_regression=True,
                                    over_param=data_config["over_param"],
                                    linear_control=True,
                                    wt_init=data_config["wt_init"],
                                    reg_hidden=data_config["reg_hidden"],
                                    ste = data_config["ste"],
                                    top_k=data_config["top_k"],
                                    detach=data_config["detach"],
                                    smax_temp= data_config["smax_temp"],
                                )
                    
                elif mname == "dgt":
                    model = DGT(in_dim=n_attributes, out_dim=n_classes, height=depth, is_regression=True, over_param=data_config["over_param"], linear_control=True, reg_hidden=data_config["reg_hidden"])
                else:
                    raise NotImplementedError
                print(model)

                ############### Extra-Experimental Settings for Ablation Study#######
                #####################################################################
                # 1. Store Model
                # torch.save(model.state_dict(), f"trained_models/ctslice_leafdist.pt")
                # 2. Load Model
                # model.load_state_dict(torch.load(f"trained_models/ctslice_leafdist.pt"))
                #####################################################################
                
                if use_gpu:
                    model = model.cuda()
                    
                loss_fn = nn.MSELoss()  # Sum of squared errors
                
                # Create dataset and dataloader
                if len(X_train) < 200:
                    # Use full batch for large datasets
                    train_dataset = TensorDataset(X_train, y_train, y_train_clf)
                    train_dataloader = DataLoader(train_dataset, batch_size=len(X_train), shuffle=True)
                    
                else:
                    # Use specified batch size for smaller datasets
                    train_dataset = TensorDataset(X_train, y_train, y_train_clf)
                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    

                # info: change here for test
                test_dataset = TensorDataset(X_test, y_test, y_test_clf) 
                test_dataloader = DataLoader(test_dataset, batch_size=len(X_test), shuffle=False)


                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                # optimizer2 = optim.Adam(model.experts.parameters(), lr=0.0001)
                

                # Learning rate scheduler
                if data_config["lr_scheduler"]:
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=data_config["lr_scheduler_gamma"])
                    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader), eta_min=0.0001,)
                else:
                    scheduler = None

                start_time = time.perf_counter()
                tr_loss = []
                te_loss = []
                tr_acc = []


                ######### 3. Training
                # FULL TRAINING LOOP
                if data_config["use_pretrained"]:
                    model.load_state_dict(torch.load(f"trained_models/{args['output_prefix']}_topk_{data_config['top_k']}_s{simulation}.pt"))
                    print('Pre-trained model loaded')
                else:
                    for epoch in range(epochs):
                        total_loss = 0.0  # Initialize total loss for the epoch
                        total_batches = 0  # Initialize total number of batches processed in the epoch

                        if args["verbose"]:
                            if epoch in [0, 1, 2, 5, 10, epochs]:
                                    device = next(model.parameters()).device  # get model's device
                                    X_test = X_test.to(device)
                                    output, _ = model(X_test, mode='test')
                                    _, topk_indices = torch.topk(model.selected_experts, k=1, dim=1)  # with topk=1
                                    selected_experts = topk_indices.flatten()
                                    expert_counts = torch.bincount(selected_experts, minlength=model.selected_experts.shape[1])
                                    expert_counts, _ = torch.sort(expert_counts, descending=True)  # sort expert counts

                                    expert_counts = expert_counts.float().cpu().numpy()
                                    print(f"Epoch: {epoch}, Leaf Dist: {expert_counts.tolist()}")
                        
                        model.train()  # Set model to training mode
                        for data, target,_ in train_dataloader:
                            if use_gpu:
                                data = data.cuda()
                                target = target.cuda()
                            # Clear gradients
                            # print(data.shape, target.shape)
                            # print(data, target)
                            output, _ = model(data, mode="train")
                            # print(output.shape, target.shape)        
                            loss = loss_fn(output.squeeze(), target)
                            
                            # L1 regularization
                            if data_config["use_L1"]:
                                l1_lambda = data_config["lamda_L1"]  # L1 regularization lambda
                                l1_regularization = torch.tensor(0., requires_grad=False)
                                if use_gpu:
                                    l1_regularization = l1_regularization.cuda()

                                # pdb: model.linear1.parameters() 
                                if data_config['name'] == 'pdb_bind':
                                    for param in model.linear1.parameters():
                                        if param.requires_grad:
                                            l1_regularization += torch.norm(param, p=1)
                                else:
                                    for param in model.parameters():
                                        if param.requires_grad:
                                            l1_regularization += torch.norm(param, p=1)
                                
                                loss += l1_lambda * l1_regularization 
                            # loss += 0.01*entropy_loss
                            
                            optimizer.zero_grad()
                            
                            # Backward pass and parameter update
                            loss.backward()
                            # Gradient clipping
                            if data_config["grad_clip"]:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01) # it was 0.01 for all experiments
                            # update weight from computed grad
                            optimizer.step()
                            
                            
                            # print(model.linear1[0].weight.grad)
                            
                            
                            # Accumulate the loss
                            total_loss += loss.item()
                            total_batches += 1
                        
                        # Optional learning rate update (if using a scheduler) for each epoch
                        if scheduler is not None:
                            scheduler.step()
                            


                        # Calculate the average loss for the epoch
                        average_loss = total_loss / total_batches
                        
                        # Print training progress (optional)
                        
                        if args["verbose"]:
                            # ABLATION: train is changes to test for ablation study
                            train_loss, multiv_acc_in = evaluate(model, train_dataloader, loss_fn, scaler, use_gpu, 'test')
                            test_loss, multiv_acc_test = evaluate(model, test_dataloader, loss_fn, scaler, use_gpu, 'test')
                            
                            print(f"Epoch: {epoch+1}/{epochs}, Average Loss: {train_loss:.4f}, Train Acc: {multiv_acc_in:.4f} Test Acc: {multiv_acc_test:.4f}")
                            
                            
                            tr_loss.append(train_loss)
                            te_loss.append(test_loss)
                            tr_acc.append(multiv_acc_in)
                    
                if args["verbose"]:
                    device = next(model.parameters()).device  # get model's device
                    X_test = X_test.to(device)
                    output, _ = model(X_test, mode='test')
                    _, topk_indices = torch.topk(model.selected_experts, k=1, dim=1)  # with topk=1
                    selected_experts = topk_indices.flatten()
                    expert_counts = torch.bincount(selected_experts, minlength=model.selected_experts.shape[1])
                    expert_counts, _ = torch.sort(expert_counts, descending=True)  # sort expert counts

                    expert_counts = expert_counts.float().cpu().numpy()
                    print(f"Epoch: Final 100%, Leaf Dist: {expert_counts.tolist()}")
                ######## 4. Fine-tune regression layer
                train_loss, multiv_acc_in = evaluate(model, train_dataloader, loss_fn, scaler, use_gpu, 'test')
                test_loss, multiv_acc_test = evaluate(model, test_dataloader, loss_fn, scaler, use_gpu, 'test')
                print(f"Before Retraining Each Leaf (Top1): Train RMSE: {multiv_acc_in:.4f}, Test RMSE: {multiv_acc_test:.4f}")
                # exit()
                if data_config["top_k"] > 0:
                    # fine tune only if the top is greater than 1
                    fine_tune_regressor(X_train, y_train, model, loss_fn, scaler, use_gpu, tr_loss, te_loss, tr_acc, mix_experts=data_config["mix_experts"])
                    
                    train_loss, multiv_acc_in = evaluate(model, train_dataloader, loss_fn, scaler, use_gpu, 'test')
                    test_loss, multiv_acc_test = evaluate(model, test_dataloader, loss_fn, scaler, use_gpu, 'test')
                    print(f"After retraining each leaf (Top1): Train RMSE: {multiv_acc_in:.4f}, Test RMSE: {multiv_acc_test:.4f}")
                    # exit()
                    #### 5. Retrain whole model
                    optimizer = optim.Adam(model.parameters(), lr=data_config["learning_rate_tune"])
                    # print('linear1 before', model.linear1[0].weight)
                    initial_weights = model.linear1[0].weight.clone()
                    for epoch in range(data_config["epochs_fine_tune"]):
                        total_loss = 0.0  # Initialize total loss for the epoch
                        total_batches = 0  # Initialize total number of batches processed in the epoch
                        
                        model.train()  # Set model to training mode
                        for data, target,_ in train_dataloader:
                            if use_gpu:
                                data = data.cuda()
                                target = target.cuda()
                            # Clear gradients
                            # print(data.shape, target.shape)
                            # print(data, target)
                            output, _ = model(data, mode="test") # full training with top1 (argmax)
                            # print(output.shape, target.shape)        
                            loss = loss_fn(output.squeeze(), target)
        
                            optimizer.zero_grad()
                            
                            # Backward pass and parameter update
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05) # it was 0.01 for all experiments
                            # update weight from computed grad
                            optimizer.step()
                            # Accumulate the loss
                            total_loss += loss.item()
                            total_batches += 1
                            # print("grad norm linear1:", model.linear1[0].weight.grad.norm().item())
                        
                        # Optional learning rate update (if using a scheduler) for each epoch
                        if scheduler is not None:
                            scheduler.step()
                        

                        # Calculate the average loss for the epoch
                        average_loss = total_loss / total_batches
                        
                        # Print training progress (optional)
                        
                        if args["verbose"]:
                            train_loss, multiv_acc_in = evaluate(model, train_dataloader, loss_fn, scaler, use_gpu, 'test')
                            test_loss, multiv_acc_test = evaluate(model, test_dataloader, loss_fn, scaler, use_gpu, 'test')
                            print(f"Epoch: {epoch+1}/{data_config['epochs_fine_tune']}, Average Loss: {train_loss:.4f}, Train Acc: {multiv_acc_in:.4f} Test Acc: {multiv_acc_test:.4f}")
                            tr_loss.append(train_loss)
                            te_loss.append(test_loss)
                            tr_acc.append(multiv_acc_in)
                    
                
                
                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                # loss_fn = nn.MSELoss()
                print("=" * 20,'Final Evaluation','=' * 20)
                train_loss, multiv_acc_in = evaluate(model, train_dataloader, loss_fn, scaler, use_gpu, 'test')
                test_loss, multiv_acc_test = evaluate(model, test_dataloader, loss_fn, scaler, use_gpu, 'test')
                ### LEAF DISTRIBUTION
                # compute leaf distirbution for train and test dataset
                train_active_leaves, train_leaf_entropy = get_leaf_distribution(model, X_train, y_train_clf, use_gpu)
                test_active_leaves, test_leaf_entropy = get_leaf_distribution(model, X_test, y_test_clf, use_gpu)
                ####
                print(f"Train RMSE: {multiv_acc_in:.4f}, Test RMSE: {multiv_acc_test:.4f}")
                print(f"Elapsed time: {elapsed_time:.2f} seconds")
                if tunning:
                    if multiv_acc_test < best_acc:
                        best_acc = multiv_acc_test
                        print(best_acc)
                        best_config = copy.deepcopy(data_config)
                        print("BEST","=" * 10)
                        print(f"Best Configuration: {best_config}")
                        print(f"Best Test RMSE: {best_acc:.4f}")
                        print("=" * 10)  

               
                
                # info: save results to file
                histories[-1].append((elapsed_time,
                                    (N),
                                    (multiv_acc_in, multiv_acc_test),
                                    (train_active_leaves, train_leaf_entropy),
                                    (test_active_leaves, test_leaf_entropy),
                                    ))
                save_histories_to_file(hconfig, histories, output_path_summ, output_path_full, command_line)

                # find the histogram of leaf distribution
                if hist_leaf_dist:
                    # save leaf distribution to file
                    os.makedirs(f"trained_models/{args['dataset']}/{mname}", exist_ok=True)
                    torch.save(model.state_dict(), f"trained_models/{args['dataset']}/{mname}/{args['output_prefix']}_sim{simulation}_seed{seed}.pt")

                    # create dir called leaf_dist if not there
                    os.makedirs(f"logs/regression/leaf_dist/{args['dataset']}/{mname}", exist_ok=True)
                    output_path_leaf_dist = f"logs/regression/leaf_dist/{args['dataset']}/{mname}/{args['output_prefix']}_leaf_dist.txt"
                    plot_path_leaf_dist = f"logs/regression/leaf_dist/{args['dataset']}/{mname}/{args['output_prefix']}_leaf_dist.png"
                    final_avg_plot_path = f"logs/regression/leaf_dist/{args['dataset']}/{mname}/{args['output_prefix']}_leaf_dist_avg.png"
                    
                    # data
                    device = next(model.parameters()).device  # get model's device
                    X_test = X_test.to(device)
                    output, _ = model(X_test, mode='test')
                    _, topk_indices = torch.topk(model.selected_experts, k=1, dim=1)  # with topk=1
                    selected_experts = topk_indices.flatten()
                    expert_counts = torch.bincount(selected_experts, minlength=model.selected_experts.shape[1])
                    expert_counts, _ = torch.sort(expert_counts, descending=True)  # sort expert counts

                    expert_counts = expert_counts.float().cpu().numpy()

                    # store leaf dist to file append it
                    with open(output_path_leaf_dist, 'a') as f:
                        f.write(f"\n--- Simulation {simulation}, Seed {seed} ---\n")
                        f.write(f"Expert Counts: {expert_counts.tolist()}\n")

                    # store running average to avg_leaf_dist
                    if 'avg_leaf_dist' not in locals():
                        avg_leaf_dist = expert_counts.astype(np.float32).copy()
                        count = 1
                    else:
                        avg_leaf_dist = (avg_leaf_dist * count + expert_counts) / (count + 1)
                        count += 1

                    # store running average to file
                    with open(output_path_leaf_dist, 'a') as f:
                        f.write(f"Running Avg Expert Counts (after {count} simulations): {avg_leaf_dist.tolist()}\n")

                    # plot the histogram
                    plt.figure(figsize=(6, 3))  
                    plt.bar(range(len(expert_counts)), expert_counts)
                    plt.xlabel('Expert Index')
                    plt.ylabel('Count')
                    plt.title(f'{mname} Sim{simulation}_{seed} {args["dataset"]}  [RMSE {multiv_acc_test}] (Total Samples: {len(X_test)})')
                    plt.xticks(range(len(expert_counts)), rotation=45, ha="right")
                    # plt.ylim(0, 3000)
                    plt.tight_layout()
                    plt.savefig(plot_path_leaf_dist, dpi=300, bbox_inches='tight')
                    plt.close()

                    # plot final average if last simulation
                    if simulation == simulations - 1:
                        plt.figure(figsize=(6, 3))  
                        plt.bar(range(len(avg_leaf_dist)), avg_leaf_dist)
                        plt.xlabel('Expert Index')
                        plt.ylabel('Average Count')
                        plt.title(f'{mname} Average Leaves Distribution ({args["dataset"]})')
                        plt.xticks(range(len(avg_leaf_dist)), rotation=45, ha="right")
                        plt.tight_layout()
                        plt.savefig(final_avg_plot_path, dpi=300, bbox_inches='tight')
                        plt.close()


                if args["verbose"]:
                    print(f"Saved summary to '{output_path_summ}'.")
                    print(f"Saved full data to '{output_path_full}'.")
    
    # save the trained model
    if not tunning:
        print("---> Saving the model")
        save_dir = f"trained_models/{args['dataset']}"
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        save_path = os.path.join(save_dir, f"{args['output_prefix']}_top{data_config['top_k']}.pt")
        torch.save(model.state_dict(), save_path)
        print("---> Model saved at", save_path)

    if tunning:
        print("=" * 70)
        print(f"Best Configuration: {best_config}")
        print(f"Best Test RMSE: {best_acc:.4f}")
        print("=" * 70)   
    