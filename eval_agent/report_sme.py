import os
import numpy as np

def compute_mean_sme(numbers):
    mean = np.mean(numbers)
    overall_std = np.std(numbers, ddof=1)

    # Calculate the square root of the sample size
    sample_size = len(numbers)
    sqrt_sample_size = np.sqrt(sample_size)
    # Calculate the standard error of the mean (SEM)
    sem = overall_std / sqrt_sample_size
    return mean, sem

def compute_mean_sme_multifile(folder_path):
    file_list = [
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    numbers = []
    for filename in file_list:
        with open(os.path.join(folder_path, filename), 'r') as file:
            try:
                num = float(file.read().strip())
                numbers.append(num)
            except ValueError:
                print(
                    f"Error: File '{filename}' does not contain a valid number."
                )

    if numbers:
        return compute_mean_sme(numbers)

    else:
        print("No valid numbers found.")
        return None


def compute_mean_singlefile(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read().strip()
            numbers = [float(num) for num in content.split()]
            return compute_mean_sme(numbers)
    except (ValueError, FileNotFoundError):
        print(f"Error: File '{file_path}' does not contain valid numbers.")
        return None


def compute_mean_sme_singlefile_withSTD(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

            numbers = []
            for line in lines:
                parts = line.strip().split('+/-')
                if len(parts) == 2:
                    try:
                        num = float(parts[0])
                        numbers.append(num)
                    except ValueError:
                        print(f"Error: Invalid number format in line: {line}")
                else:
                    print(f"Error: Invalid line format: {line}")

            if numbers:
                return compute_mean_sme(numbers)
            else:
                print("No valid numbers found.")
                return None
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

ENV_TYPE = 'lunar'
AGENT_TYPE = 'fcnn'
LOG_NAME = 'aaai_32x32a_32x32c_v3_rand'  #logs/lunar_cpu/fcnn/aaai_32x32a_32x32c_v3_rand
DEVICE = 'cpu'

#####Reported Model: Acrobot
# ICCT: cpu/aaai_15n_32x32c_rand
# DGT: cpu/aaai_15n_32x32c_rand
# ProLoNet: cpu/aaai_15na_soft_32x32c_rand
# FCNN: val 32x32
# 16x16: cpu/16x16_32x32c_rand
# 16 linear: TODO
# 16+relu: cpu/16x_32x32c_rand
# DTNet: cpu/aaai_15n_32x32c_rand
#####

#####Reported Model: Cart Pole
# ICCT: cpu/'aaai_15n_16x16c_rand'
# DGT:
# ProLoNet: cpu/'aaai_15na_soft_16x16c_rand'
# FCNN: val 16x16
# 16x16: cpu/16l_16x16c_rand
# 16 linear: cpu/16xa_16x16c_rand
# 16+relu: cpu/16x16_16x16c_rand
# DTNet: cpu/aaai_15n_16x16c_rand
#####


#####Reported Model: Lunar Lander
# ICCT:
# hard: 'aaai_32la_32x32c_rand'
# soft (prolonet): TODO
# DGT: cpu/aaai_31n_32x32c_rand
# ProLoNet: fixed_nodup_rand
# FCNN:
# 32+relu, v32x32: TODO
# 32linear, v32x32: aaai_32lin_32x32c_v3_rand
# 32x32, v32x32:  cpu/'aaai_32x32a_32x32c_v3_rand'
# DTNet: cpu/aaai_31n_32x32c_v2_rand
#####

folder_path = f'logs/{ENV_TYPE}_{DEVICE}/{AGENT_TYPE}/{LOG_NAME}/eval'  # Replace with the actual folder path
mean, sme = compute_mean_sme_multifile(folder_path)

if mean is not None:
    print(f"ENV_TYPE: {ENV_TYPE}; AGENT_TYPE: {AGENT_TYPE}; LOG_NAME: {LOG_NAME}")
    print(f"Mean of numbers in folder: {mean:.2f} +/- {sme:.2f}")
