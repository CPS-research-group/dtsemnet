"""Various functions to test the agent at the end of training
test_single_episode: test agent in environment for a single episode
running_avg: calculate running average of rewards over 100 episodes
"""

import numpy as np
import torch

# Evaluation function
def evaluate(model, dataloader, loss_fn, scaler, use_gpu=False, mode='test'):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total_samples = 0
    total_squared_error = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for data, target,_ in dataloader:
            if scaler is not None:
                target = torch.tensor(scaler.inverse_transform(target.reshape(-1, 1)).reshape(-1), dtype=torch.float32)
            batch_size = data.size(0)
            total_samples += batch_size
            if use_gpu:
                data = data.cuda()
                target = target.cuda()
            output, _ = model(data, mode=mode)
            if scaler is not None:
                output = torch.tensor(scaler.inverse_transform(output.detach().cpu().numpy()), dtype=torch.float32)
            else:
                output = torch.tensor(output.detach().cpu().numpy(), dtype=torch.float32)
            if use_gpu:
                output = output.cuda()
            loss = loss_fn(output.squeeze(), target)
            total_loss += loss.item() * batch_size
            total_squared_error += ((output.squeeze() - target) ** 2).sum().item()

    mean_squared_error = total_squared_error / total_samples
    rmse = np.sqrt(mean_squared_error)
    return total_loss / total_samples, rmse

def get_leaf_distribution(model, X_train, y_train, use_gpu=False):
    """
    Get the distribution of leaves in the model.
    """
    model.eval()  # Set model to evaluation mode
    if use_gpu:
        X_train = X_train.cuda()
        y_train = y_train.cuda()
    
    output, _ = model(X_train, mode='test')
    
    # Get the top-1 expert for each sample
    _, topk_indices = torch.topk(model.selected_experts, k=1, dim=1)
    selected_experts = topk_indices.flatten()
    
    # Count the occurrences of each leaf
    num_leaves = model.selected_experts.shape[1]
    expert_counts = torch.bincount(selected_experts, minlength=num_leaves)
    total_samples = expert_counts.sum().item()
    probs = expert_counts / total_samples

    # 1. Coverage: number of leaves getting at least 1% of the samples
    threshold = 0.01
    num_active_leaves = (probs >= threshold).sum().item()

    # 2. Normalized entropy (0 to 1 scale)
    eps = 1e-12  # avoid log(0)
    entropy = -torch.sum(probs * torch.log(probs + eps)).item()
    max_entropy = np.log(num_leaves)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    return num_active_leaves, normalized_entropy
    
    

def running_avg(arr, num_seeds=5, N=100):
    """Calculate running average of rewards over 100 episodes
    
    Arguments:
        arr {list} -- list of rewards for each seed
        num_seeds {int} -- number of seeds

    Returns:
        np.array -- array of running average rewards for each seed
    """
    run_avg_arr = []
    for s in range(num_seeds):
        run_avg = [arr[s][0]]
        for i in range(len(arr[s]) - 1):
            if i < N: # for previous 100 steps
                ix = i + 1
                ra = sum(arr[s][0:ix]) / float(len(arr[s][0:ix]))

            else:
                ix = i + 1
                ra = sum(arr[s][ix-int(N):ix]) / N

            # print(ra)
            run_avg.append(ra)
        run_avg_arr.append(run_avg)

    run_avg_arr = np.array(run_avg_arr)
    return run_avg_arr


def get_confidence_interval(arr):
    """Calculate confidence interval for array of rewards.
    
    Arguments:
        arr {np.array} -- array of rewards
    
    Returns:
        tuple -- mean, lower bound, upper bound
    """
    mean = np.mean(arr, axis=0)
    std = np.std(arr, ddof=1, axis=0) / np.sqrt(arr.shape[0])
    cil, cih = mean - std, mean + std
    return mean, cil, cih