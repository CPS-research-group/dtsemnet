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
from torch.autograd import Variable
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
        string_summ += f"Average in-sample multivariate accuracy: {'{:.4f}'.format(np.mean(multiv_acc_in))} ± {'{:.4f}'.format(np.std(multiv_acc_in))}\n"
        string_summ += f"Average test multivariate accuracy: {'{:.4f}'.format(np.mean(multiv_acc_test))} ± {'{:.4f}'.format(np.std(multiv_acc_test))}\n"
        string_summ += "\n"
        string_summ += f"Best test multivariate accuracy: {'{:.4f}'.format(multiv_acc_test[np.argmax(multiv_acc_test)])}\n"
        string_summ += f"Average elapsed time: {'{:.4f}'.format(np.mean(elapsed_times))} ± {'{:.4f}'.format(np.std(elapsed_times))}\n"

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
class NN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        hidden = 128
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_size)
        )
        # Orthogonal initialization
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)

    def forward(self, x):
        return self.layers(x)


# Evaluation function
def evaluate(model, dataloader, loss_fn, use_gpu):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for data, target in dataloader:
            batch_size = data.size(0)
            total += batch_size
            if use_gpu:
                data = data.cuda()
                target = target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item()*batch_size
            _, predicted = torch.max(output.data, 1)
            correct += predicted.eq(target).sum().item()

    return total_loss / total, correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DT*-Net training script.')
    parser.add_argument('-i', '--dataset', help="What dataset to use?", required=True, type=str)
    parser.add_argument('-m', '--model', help="Which model to use?", required=True, type=str)
    parser.add_argument('-s', '--simulations', help="How many simulations?", required=False, type=int, default=1)
    parser.add_argument('--output_prefix', help='Which output name to use?', required=False, default="dtsemnet", type=str)
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
        from src.sup_configs_dtsemnet2 import get_config, load_dataset, real_dataset_list
    elif mname == "dtsemnet":
        from src.sup_configs_dtsemnet2 import get_config, load_dataset, real_dataset_list
    
    

    command_line = str(args)
    command_line += "\n\npython -m large_dataset.dtsemnet_train " + " ".join(
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
    
    # info: Loop over each dataset (same conficuration as Original)
    for dataset_id, data_config in enumerate(data_configs):
        print(data_config)
        train_dataloader, val_dataloader, test_dataloader = load_dataset(data_config) 
        
        N = len(train_dataloader.dataset)

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
                model = NN(n_attributes, n_classes)
            elif mname == "dtsemnet":
                # model = DTSemNet(
                #                 in_dim=n_attributes,
                #                 out_dim=n_classes,
                #                 height=depth,
                #                 is_regression=False,
                #                 over_param=[],
                #                 linear_control=False,
                #                 wt_init=True,
                #             )
                model = DTSemNet(
                                in_dim=n_attributes,
                                out_dim=n_classes,
                                height=depth,
                                is_regression=False,
                                over_param=data_config["over_param"],
                                linear_control=False,
                                wt_init=data_config["wt_init"],
                            )
            elif mname == "dgt":
                model = DGT(in_dim=n_attributes, out_dim=n_classes, height=depth, is_regression=True, over_param=[])
            else:
                raise NotImplementedError
            
            print(model)
            
            
            if use_gpu:
                model = model.cuda()
            


            # Calculate the number of trainable parameters in the model
            # num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # print(f"Number of trainable parameters: {num_trainable_params}")
            
            if use_gpu:
                loss_fn = nn.CrossEntropyLoss().cuda()
            else:
                loss_fn = nn.CrossEntropyLoss()

            # Define optimizer and scheduler (optional)
            # MNIST: None
            
            
            if data_config["opt"] == "sgd":
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=data_config["momentum"], weight_decay=0.0005, nesterov=True)
            elif data_config["opt"] == "adam":
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            elif data_config["opt"] == "rmsprop":
                print('using RMSprop optimizer')
                optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=data_config["momentum"], eps=1e-05)

            # Learning rate scheduler
            if data_config["lr_scheduler"]:
                if data_config['lr_scheduler_type'] == 'linear':
                    print('using linear scheduler')
                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=data_config["lr_scheduler_gamma"])
                else:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_dataloader), eta_min=0.0001, )
            else:
                scheduler = None

            start_time = time.perf_counter()
            # FULL TRAINING LOOP
            for epoch in range(epochs):
                total_loss = 0.0  # Initialize total loss for the epoch
                total_batches = 0  # Initialize total number of batches processed in the epoch
                
                for data, target in train_dataloader:
                    
                    
                    if use_gpu:
                        data = data.cuda()
                        target = target.cuda()
                    # Clear gradients
                    optimizer.zero_grad()
                    data, target = Variable(data), Variable(target)
                    # Forward pass

                    output = model(data)
                   
                     
                    loss = loss_fn(output, target)
                    
                    # L1 regularization
                    if data_config["use_L1"]:
                        l1_lambda = data_config["lamda_L1"]  # L1 regularization lambda
                        l1_regularization = torch.tensor(0., requires_grad=False)
                        if use_gpu:
                            l1_regularization = l1_regularization.cuda()
                        for param in model.parameters():
                            if param.requires_grad:
                                l1_regularization += torch.norm(param, p=1)       
                        loss += l1_lambda * l1_regularization

                    # Backward pass and parameter update
                    loss.backward()
                    # Gradient clipping
                    if data_config["grad_clip"]:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
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
                    test_loss, multiv_acc_test = evaluate(model, test_dataloader, loss_fn, use_gpu)
                    print(f"Epoch: {epoch+1}/{epochs}, Average Loss: {average_loss:.4f}, tloss:{test_loss:.4f}, Test Acc: {multiv_acc_test:.4f}")


            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            # loss_fn = nn.MSELoss()
            train_loss, multiv_acc_in = evaluate(model, train_dataloader, loss_fn, use_gpu)
            test_loss, multiv_acc_test = evaluate(model, test_dataloader, loss_fn, use_gpu)
            print(f"Train RMSE: {multiv_acc_in:.4f}, Test RMSE: {multiv_acc_test:.4f}")
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
        
        # Open the file in read mode
        with open(output_path_summ, 'r') as file:
            # Read and print the contents
            print(file.read())
            
