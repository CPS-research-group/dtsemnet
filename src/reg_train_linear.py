import pdb
import math
import argparse
import copy
import sys
from datetime import datetime
import json
import pickle


import time
import numpy as np


from rich import print


import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
torch.backends.cudnn.deterministic = True
## DTSemNet
from src.dtsemnet import DTSemNet
from src.dgt import DGT
##



def save_histories_to_file(configs, histories, output_path_summary, output_path_full, prefix=""):
    string_summ = prefix + "\n"
    string_full = prefix + "\n"
    for config, history in zip(configs, histories):
        elapsed_times, N, multiv_info = zip(*history)
        multiv_acc_in, multiv_acc_test = zip(*multiv_info)
        

        string_summ += "--------------------------------------------------\n\n"
        string_summ += f"DATASET: {config['name']}\n"
        string_summ += f"Num Total Samples: {N[0]}\n"
        string_summ += f"{len(elapsed_times)} simulations executed. Seed: {config['tseed']}\n"
        string_summ += f"Average in-sample multivariate accuracy: {'{:.3f}'.format(np.mean(multiv_acc_in))} ± {'{:.3f}'.format(np.std(multiv_acc_in))}\n"
        string_summ += f"Average test multivariate accuracy: {'{:.3f}'.format(np.mean(multiv_acc_test))} ± {'{:.3f}'.format(np.std(multiv_acc_test))}\n"
        string_summ += "\n"
        string_summ += f"Best test multivariate accuracy: {'{:.3f}'.format(multiv_acc_test[np.argmax(multiv_acc_test)])}\n"
        string_summ += f"Average elapsed time: {'{:.3f}'.format(np.mean(elapsed_times))} ± {'{:.3f}'.format(np.std(elapsed_times))}\n"

        string_full += "--------------------------------------------------\n\n"
        string_full += f"DATASET: {config['name']}\n"

        for (elapsed_time, \
             (N), \
             (multiv_acc_in, multiv_acc_test)) in history:
            string_full += f"In-sample:" + "\n"
            string_full += f"        Multivariate accuracy: {multiv_acc_in}" + "\n"
            string_full += f"Test:" + "\n"
            string_full += f"        Multivariate accuracy: {multiv_acc_test}" + "\n"
            string_full += f"Elapsed time: {elapsed_time}" + "\n"
            
            string_full += "\n\n--------\n\n"

    with open(output_path_summary, "w", encoding="utf-8") as text_file:
        text_file.write(string_summ)

    with open(output_path_full, "w", encoding="utf-8") as text_file:
        text_file.write(string_full)

# MLP Model Definition
class RegNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        hidden = 64
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

    def forward(self, x):
        return self.layers(x)


# Evaluation function
def evaluate(model, dataloader, loss_fn, scaler, use_gpu=False):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total_samples = 0
    total_squared_error = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for data, target in dataloader:
            target = torch.tensor(scaler.inverse_transform(target.reshape(-1, 1)).reshape(-1), dtype=torch.float32)
            batch_size = data.size(0)
            total_samples += batch_size
            if use_gpu:
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            output = torch.tensor(scaler.inverse_transform(output.detach().cpu().numpy()), dtype=torch.float32)
            if use_gpu:
                output = output.cuda()
            loss = loss_fn(output.squeeze(), target)
            total_loss += loss.item() * batch_size
            total_squared_error += ((output.squeeze() - target) ** 2).sum().item()

    mean_squared_error = total_squared_error / total_samples
    rmse = np.sqrt(mean_squared_error)
    return total_loss / total_samples, rmse


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
    

    # load hyper parameters according to mname
    if mname == "mlp":
        from src.sup_configs_dtregnet_linear import get_config, load_dataset, real_dataset_list
    elif mname == "dtregnet":
        from src.sup_configs_dtregnet_linear import get_config, load_dataset, real_dataset_list
    elif mname == "dgt":
        from src.sup_configs_dgt_linear import get_config, load_dataset, real_dataset_list

    

    command_line = str(args)
    command_line += "\n\npython -m reg_dataset_linear.reg_train_linear " + " ".join(
        [f"--{key} {val}" for (key, val) in args.items()]) + "\n\n---\n\n"
    curr_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    # todo: change output file name
    output_path_summ = f"results/{mname}_{args['output_prefix']}_{curr_time}_summary.txt"
    output_path_full = f"results/{mname}_{args['output_prefix']}_{curr_time}_full.txt"

    
    dataset_list = real_dataset_list
    

    if args['dataset'].endswith('all'):
        data_configs = [get_config(d) for d in dataset_list]
    elif args['dataset'].endswith("onwards"):
        dataset_start = dataset_list.index(args['dataset'][:-len("_onwards")])
        data_configs = [get_config(d) for d in dataset_list[dataset_start:]]
    else:
        data_configs = [get_config(args['dataset'])]

    # info:
        # data_configs: list of dictionaries, each dictionary contains information about dataset
    


    histories = []
    hconfig = []
    
    
    # info: Loop over each dataset (same conficuration as Original)
    for dataset_id, data_config in enumerate(data_configs):
       for seed in range(data_config["nseed"]): 
            data_config["tseed"] = seed+1
            hconfig.append(data_config.copy())
            
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
            

            # Hyperparameters
            learning_rate = data_config["learning_rate"]
            batch_size = data_config["batch_size"]
            epochs = data_config["epochs"]
            
            
            # info: list of histories for each dataset [[h_ds1], [h_ds2], []]
            histories.append([]) 

            # dtsement does not support more classes than leaves
            if mname == 'dtsemnet' and n_classes > n_leaves:
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
                elif mname == "dtregnet":
                    model = DTSemNet(
                                    in_dim=n_attributes,
                                    out_dim=n_classes,
                                    height=depth,
                                    is_regression=True,
                                    over_param=data_config["over_param"],
                                    linear_control=True,
                                    wt_init=False,
                                    reg_hidden=data_config["reg_hidden"],
                                )
                elif mname == "dgt":
                    model = DGT(in_dim=n_attributes, out_dim=n_classes, height=depth, is_regression=True, over_param=data_config["over_param"], linear_control=True, reg_hidden=data_config["reg_hidden"])
                else:
                    raise NotImplementedError
                print(model)
                
                if use_gpu:
                    model = model.cuda()

                # Calculate the number of trainable parameters in the model
                # num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                # print(f"Number of trainable parameters: {num_trainable_params}")
                # exit(0)
                loss_fn = nn.MSELoss()  # Sum of squared errors

                

                # Create dataset and dataloader
                if len(X_train) < 200:
                    # Use full batch for large datasets
                    train_dataset = TensorDataset(X_train, y_train)
                    train_dataloader = DataLoader(train_dataset, batch_size=len(X_train), shuffle=True)
                else:
                    # Use specified batch size for smaller datasets
                    train_dataset = TensorDataset(X_train, y_train)
                    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                # info: change here for test
                test_dataset = TensorDataset(X_test, y_test) 
                test_dataloader = DataLoader(test_dataset, batch_size=len(X_test), shuffle=False)
                # Define optimizer and scheduler (optional)
                
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                # optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=0.8, eps=1e-5)
                # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0, weight_decay=0.00001)

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
                # FULL TRAINING LOOP
                for epoch in range(epochs):
                    total_loss = 0.0  # Initialize total loss for the epoch
                    total_batches = 0  # Initialize total number of batches processed in the epoch
                    
                    for data, target in train_dataloader:
                        if use_gpu:
                            data = data.cuda()
                            target = target.cuda()
                        # Clear gradients
                        
                        
                        # Forward pass

                        output = model(data)
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
                        
                        optimizer.zero_grad()
                        # Backward pass and parameter update
                        loss.backward()
                        # Gradient clipping
                        if data_config["grad_clip"]:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
                            # 0.05 for all
                        
                        optimizer.step()
                        
                        # Accumulate the loss
                        total_loss += loss.item()
                        total_batches += 1
                    
                    # Optional learning rate update (if using a scheduler)
                    if scheduler is not None:
                        scheduler.step()
                    
                    # Calculate the average loss for the epoch
                    average_loss = total_loss / total_batches
                    
                    # Print training progress (optional)
                    
                    if args["verbose"]:
                        train_loss, multiv_acc_in = evaluate(model, train_dataloader, loss_fn, scaler, use_gpu)
                        test_loss, multiv_acc_test = evaluate(model, test_dataloader, loss_fn, scaler, use_gpu)
                        print(f"Epoch: {epoch+1}/{epochs}, Average Loss: {train_loss:.4f}, Train Acc: {multiv_acc_in:.4f} Test Acc: {multiv_acc_test:.4f}")
                        tr_loss.append(train_loss)
                        te_loss.append(test_loss)
                        tr_acc.append(multiv_acc_in)


                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                # loss_fn = nn.MSELoss()
                train_loss, multiv_acc_in = evaluate(model, train_dataloader, loss_fn, scaler, use_gpu)
                test_loss, multiv_acc_test = evaluate(model, test_dataloader, loss_fn, scaler, use_gpu)
                print(f"Train RMSE: {multiv_acc_in:.4f}, Test RMSE: {multiv_acc_test:.4f}")
                print(f"Elapsed time: {elapsed_time:.2f} seconds")

               
                
                # info: save results to file
                histories[-1].append((elapsed_time,
                                    (N), \
                                    (multiv_acc_in, multiv_acc_test)
                                    ))
                save_histories_to_file(hconfig, histories, output_path_summ, output_path_full, command_line)

                if args["verbose"]:
                    print(f"Saved summary to '{output_path_summ}'.")
                    print(f"Saved full data to '{output_path_full}'.")
            
        # Open the file in read mode
    with open(output_path_summ, 'r') as file:
        # Read and print the contents
        print(file.read())

    # Read the content of the module file
    with open('src/sup_configs_dtregnet_linear.py', 'r') as module_file:
        module_content = module_file.read()

    # Append the content of the module file to the output file
    with open(output_path_summ, 'a') as output_file:
        output_file.write("\n========\n")
        output_file.write(module_content)
