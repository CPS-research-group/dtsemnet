import pdb
import math
import argparse
import copy
import sys
from datetime import datetime
import json
import pickle

from sklearn.model_selection import train_test_split
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
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
        string_summ += f"{len(elapsed_times)} simulations executed.\n"
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
class YourModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, output_size)
        )

    def forward(self, x):
        return self.layers(x)


# Evaluation function
def evaluate(model, dataloader, loss_fn):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct = 0
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for data, target in dataloader:
            output = model(data)
            total_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # Get predictions
            correct += pred.eq(target.view_as(pred)).sum().item()

    return total_loss / len(dataloader.dataset), correct / len(dataloader.dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DT*-Net training script.')
    parser.add_argument('-i', '--dataset', help="What dataset to use?", required=True, type=str)
    parser.add_argument('-m', '--model', help="Which model to use?", required=True, type=str)
    parser.add_argument('-d', '--depth', help="Depth of tree", required=True, type=int)
    parser.add_argument('-s', '--simulations', help="How many simulations?", required=False, type=int, default=1)
    parser.add_argument('--output_prefix', help='Which output name to use?', required=False, default="dtsemnet", type=str)
    parser.add_argument('--should_normalize_dataset', help='Should normalize dataset?', required=False, default=True,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--verbose', help='Is verbose?', required=False, default=True,
                        type=lambda x: (str(x).lower() == 'true'))
    args = vars(parser.parse_args())

    # Initialization
    depth = args["depth"]
    simulations = args["simulations"]
    n_inners, n_leaves = 2 ** depth - 1, 2 ** depth
    mname = args['model'] # model name

    # load hyper parameters according to mname
    if mname == "mlp":
        pass
    elif mname == "dtsemnet":
        from src.sup_configs_dtsemnet import get_config, load_dataset, artificial_dataset_list, real_dataset_list
    elif mname == "dgt":
        from src.sup_configs_dgt import get_config, load_dataset, artificial_dataset_list, real_dataset_list
    elif mname == "cart":
        from src.sup_configs import get_config, load_dataset, artificial_dataset_list, real_dataset_list

    command_line = str(args)
    command_line += "\n\npython -m small_dataset.dtsemnet_train " + " ".join(
        [f"--{key} {val}" for (key, val) in args.items()]) + "\n\n---\n\n"
    curr_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    # todo: change output file name
    output_path_summ = f"results/{mname}_{args['output_prefix']}_depth{args['depth']}_{curr_time}_summary.txt"
    output_path_full = f"results/{mname}_{args['output_prefix']}_depth{args['depth']}_{curr_time}_full.txt"

    if args['dataset'].startswith("artificial"):
        dataset_list = artificial_dataset_list
    else:
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
    # info: Loop over each dataset (same conficuration as Original)
    for dataset_id, data_config in enumerate(data_configs):
        X, y = load_dataset(data_config)
        N = len(X)
        n_attributes = data_config["n_attributes"]
        n_classes = data_config["n_classes"]

        

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


        
            # todo: check loading dataset and test train split
            # why: why 25% of data is thrown away? Yes, consider that to be validation dataset
            if args["dataset"].startswith("artificial"):
                X_train, y_train = X, y
                X_test, y_test = X, y
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=simulation, stratify=y)
                X_test, _, y_test, _ = train_test_split(X_test, y_test, test_size=0.5, random_state=simulation, stratify=y_test)

            # todo: check normalization of dataset
            if args["should_normalize_dataset"]:
                scaler = StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
            else:
                scaler = None


            # todo: train information for each dataset
            ### TRAIN ###
            print("=" * 70)
            print(
                f"[green]Iteration {simulation}/{args['simulations']}[/green] [red](dataset: {data_config['name']}, {dataset_id}/{len(data_configs)}):[/red] [yellow] Num Sample: {N} [/yellow]")
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
                pass
            elif mname == "dtsemnet":
                if data_config["code"] == 'avila':
                    model = DTSemNet(
                                in_dim=n_attributes,
                                out_dim=n_classes,
                                height=depth,
                                is_regression=False,
                                over_param=[],
                                wt_init=False,
                                custom_leaf=[ 0, 1, 4, 0, 2, 5, 6, 7, 8, 9, 10, 11, 0, 1, 4, 8],
                            )
                    # [ 0, 1, 4, 0, 2, 5, 6, 7, 8, 9, 10, 11, 0, 1, 4, 8]
                    
                elif data_config["code"] == 'banknote' and depth == 3:
                    model = DTSemNet(
                                in_dim=n_attributes,
                                out_dim=n_classes,
                                height=depth,
                                is_regression=False,
                                over_param=[],
                                custom_leaf=[3, 1, 0, 1, 0, 1, 2, 0],
                            )
                else:
                    model = DTSemNet(
                                    in_dim=n_attributes,
                                    out_dim=n_classes,
                                    height=depth,
                                    is_regression=False,
                                    over_param=[],
                                    wt_init=False,
                                    custom_leaf=None,
                                )

            elif mname == "dgt":
                model = DGT(in_dim=n_attributes, out_dim=n_classes, height=depth, is_regression=False, over_param=[], linear_control=False, reg_hidden=0)
            else:
                raise NotImplementedError

            print(model)
            loss_fn = nn.CrossEntropyLoss()  # Replace with your loss function if needed

            # Prepare data as tensors
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.long)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.long)

            # Create dataset and dataloader
            if len(X_train) < 400:
                # Use full batch for small datasets
                train_dataset = TensorDataset(X_train, y_train)
                train_dataloader = DataLoader(train_dataset, batch_size=len(X_train), shuffle=True)
            else:
                # Use specified batch size for smaller datasets
                train_dataset = TensorDataset(X_train, y_train)
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            test_dataset = TensorDataset(X_test, y_test)
            test_dataloader = DataLoader(test_dataset, batch_size=len(X_test), shuffle=False)
            # Define optimizer and scheduler (optional)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            # Learning rate scheduler
            if data_config["lr_scheduler"]:
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=data_config["lr_scheduler_gamma"])
            else:
                scheduler = None

            start_time = time.perf_counter()
            # FULL TRAINING LOOP
            for epoch in range(epochs):
                total_loss = 0.0  # Initialize total loss for the epoch
                total_batches = 0  # Initialize total number of batches processed in the epoch
                
                for data, target in train_dataloader:
                    # Clear gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    output = model(data)
                    
                    
                    loss = loss_fn(output, target)

                    # L1 regularization
                    if data_config["use_L1"]:
                        if mname == "dtsemnet":
                            l1_lambda = data_config["lamda_L1"]  # L1 regularization lambda
                            l1_regularization = torch.tensor(0., requires_grad=False)
                            for param in model.linear1.parameters():
                                l1_regularization += torch.norm(param, p=1)       
                            loss += l1_lambda * l1_regularization

                        elif mname == "dgt":
                            l1_lambda = data_config["lamda_L1"]  # L1 regularization lambda
                            l1_regularization = torch.tensor(0., requires_grad=False)
                            for param in model._predicate_l.parameters():
                                l1_regularization += torch.norm(param, p=1)       
                            loss += l1_lambda * l1_regularization
                    
                    # Backward pass and parameter update
                    loss.backward()
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
                    test_loss, multiv_acc_test = evaluate(model, test_dataloader, loss_fn)
                    print(f"Epoch: {epoch+1}/{epochs}, Average Loss: {average_loss:.4f}")


            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            train_loss, multiv_acc_in = evaluate(model, train_dataloader, loss_fn)
            test_loss, multiv_acc_test = evaluate(model, test_dataloader, loss_fn)
            print(f"Train Accuracy: {multiv_acc_in:.4f}, Test Accuracy: {multiv_acc_test:.4f}")
            print(f"Elapsed time: {elapsed_time:.2f} seconds")

            

            # info: save results to file
            histories[-1].append((elapsed_time,
                                (N), \
                                (multiv_acc_in, multiv_acc_test)
                                ))
            save_histories_to_file(data_configs, histories, output_path_summ, output_path_full, command_line)

            if args["verbose"]:
                print(f"Saved summary to '{output_path_summ}'.")
                print(f"Saved full data to '{output_path_full}'.")
            
            
            
            # ##===========> Extra analysis
            # # Print all trainable model weights
            # farray = model._or_l.weight.detach().numpy()
            # # print(farray)
            # lassign = np.argmax(farray, axis=0)
            # print(lassign)
        
        # Open the file in read mode
        with open(output_path_summ, 'r') as file:
            # Read and print the contents
            print(file.read())
        

        