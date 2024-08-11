import numpy as np
import scipy.stats as stats
import scikit_posthocs as sp

# Mean accuracies of 5 models across 5 datasets (example data)
# mean_accuracies = np.array([
#     [0.6860, 0.8755, 0.9610, 0.9702, 0.8203, 0.9616, 0.8429, 0.8919],  # Model 1
#     [.6780, .8664, .9586, .9636, .7952, .94, .8367, .8613],  # Model 2
#     [.6841, .8741, .9501, .9608, .8121, .9505, .8252, .8741],  # Model 3
#     [.5753, .8418, .9423, .8994, .7403, .8559, .7831, .7013],  # Model 4  
# ])

# mean_accuracies = np.array([
#     [90.2, 99.8, 78.5, 100, 100, 93.3, 97.2, 62.2, 58.6, 53.5, 91.4, 92.9, 82.1, 93.3],  # Model 1
#     [88.6, 99.8, 78.3, 100, 100, 92.1, 97.2, 59.7, 56.6, 52.1, 89, 92.4, 80.8, 91.9],  # Model 2
#     [77.4, 96.6, 76.9, 99.7, 99, 84.5, 94.7, 55.8, 56.9, 52.3, 83.2, 90.6, 70.9, 64.6],  # Model 3
#     [74.9, 93.6, 77.1, 100, 99, 84.3, 94.7, 54, 55.9, 52, 80.5, 91.8, 70.6, 53.2],  # Model 4  
#     [77.8, 95.2, 76.1, 100, 100, 86.1, 95.5, 59.6, 55.8, 51.4, 77.9, 91.5, 71.7, 65.2] # model 5
# ])

mean_accuracies = np.array([
    [62.2, 58.6, 53.5, 91.4, 82.1, 93.3],  # Model 1
    [59.7, 56.6, 52.1, 89, 80.8, 91.9],  # Model 2
    [55.8, 56.9, 52.3, 83.2, 70.9, 64.6],  # Model 3
    [54, 55.9, 52, 80.5, 70.6, 53.2],  # Model 4  
    [59.6, 55.8, 51.4, 77.9, 71.7, 65.2] # model 5
])

# Transpose the mean_accuracies array to fit the required input shape (datasets as columns, models as rows)
data = mean_accuracies.T / 100

num_datasets, num_methods = data.shape
print("Methods:", num_methods, "Datasets:", num_datasets)
# Perform the Friedman test
friedman_statistic, p_value = stats.friedmanchisquare(*data)
print(f"Friedman test statistic: {friedman_statistic}, p-value: {p_value}")

# If the p-value is significant, perform the Nemenyi post-hoc test
if p_value < 0.05:
    nemenyi_results = sp.posthoc_nemenyi_friedman(data)
    print("Nemenyi post-hoc test results:\n", nemenyi_results)
else:
    print("The p-value is not significant; no need for post-hoc test.")



import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot_cd_diagram(ranks, cd, labels):
        plt.figure(figsize=(10, 2))
        plt.hlines(1, 0, len(labels) + 1, colors='black')
        for i, rank in enumerate(ranks):
            plt.plot(rank, 1, 'o', label=labels[i])
        plt.xticks(np.arange(1, len(labels) + 1), labels)
        plt.xlabel('Average Rank')
        plt.title(f'Critical Difference (CD) = {cd:.3f}')
        # plt.legend(loc='u')
        plt.savefig('/home/subratpr001/Documents/neurodt/dtsemnet-ecai/results/cd_diagram.png')
        plt.show()

# Calculate average ranks
ranks = np.array([stats.rankdata(-row, method='average') for row in mean_accuracies.T])
average_ranks = np.mean(ranks, axis=0)

# Number of models and datasets
k = mean_accuracies.shape[0]
N = mean_accuracies.shape[1]

print(ranks)
print(average_ranks)
print(k)
print(N)

# Calculate the Critical Difference
q_alpha = 2.569  # q_alpha for Nemenyi test with alpha = 0.05 and k models
cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))

# Plot the CD diagram
# plot_cd_diagram(average_ranks, cd, ['Model 1', 'Model 2', 'Model 3', 'Model 4'])


## from gemini
# import numpy as np
# import scipy.stats as stats
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Example mean accuracies of 5 models across 5 datasets
# mean_accuracies = np.array([
#     [0.6860, 0.8755, 0.9610, 0.9702, 0.8203, 0.9616, 0.8429, 0.8919],  # Model 1
#     [.6780, .8664, .9586, .9636, .7952, .94, .8367, .8613],  # Model 2
#     [.6841, .8741, .9501, .9608, .8121, .9505, .8252, .8741],  # Model 3
#     [.5753, .8418, .9423, .8994, .7403, .8559, .7831, .7013],  # Model 4  
# ])

# # Transpose the mean_accuracies array
# data = mean_accuracies.T

# # Perform the Friedman test
# friedman_statistic, p_value = stats.friedmanchisquare(*data)
# print(f"Friedman test statistic: {friedman_statistic}, p-value: {p_value}")

# if p_value < 0.05:
#     # Calculate average ranks with ties handled correctly
#     ranks = np.array([stats.rankdata(-row, method='average') for row in mean_accuracies.T])
#     average_ranks = np.mean(ranks, axis=1)

#     # Number of models and datasets
#     k = mean_accuracies.shape[0]
#     N = mean_accuracies.shape[1]

#   # Calculate the Critical Difference
#     q_alpha = 2.569  # q_alpha for Nemenyi test with alpha = 0.05 and k models
#     cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))

#     # Perform Nemenyi post-hoc test manually
#     nemenyi_results = []
#     for i in range(k):
#         for j in range(i + 1, k):
#             difference = abs(average_ranks[i] - average_ranks[j])
#             is_significant = difference >= cd
#             nemenyi_results.append((f"Model {i+1}", f"Model {j+1}", is_significant))

#     print("Nemenyi post-hoc test results:")
#     for model1, model2, is_significant in nemenyi_results:
#         print(f"{model1} vs {model2}: {is_significant}")