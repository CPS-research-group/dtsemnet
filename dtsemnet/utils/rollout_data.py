import numpy as np
import random
import torch


def get_rollout_data(env, model, num_samples, prolo=False):
    """Generate rollout data"""
    # create original data
    original_data = []
    num_sam = 0
    for seed in range(10000):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        env.seed(seed)

        state = env.reset()
        done = False
        while not done:
            if prolo:
                action = model.get_action(state)
            else:
                action = model.predict(state, deterministic=True)[0]
            state, reward, done, info = env.step(action)
            if not done:
                original_data.append(state)
                num_sam += 1
                if num_sam == num_samples:
                    original_data = torch.tensor(original_data,
                                                 dtype=torch.float32)
                    assert len(
                        original_data
                    ) == num_samples, "original data not the right size with num_samples"
                    return original_data


def perturb_data(original_data, error, dim=6):
    """Perturb the original data by adding noise to the state"""
    # Generate perturbed data
    perturbed_data = original_data.clone().detach()

    for state in perturbed_data:
        # === random vector
        # Generate random noise vector
        noise_vector = np.random.uniform(size=dim)
        # Scale the noise vector to have L1 magnitude of 0.05
        current_magnitude = np.sum(np.abs(noise_vector))
        # current_magnitude = np.linalg.norm(noise_vector, ord=2)
        
        scaled_noise_vector = noise_vector * (error /
                                              current_magnitude)
        state[:dim] += scaled_noise_vector

    return perturbed_data
