import re

import os

def list_txt_files(directory):
    txt_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.txt')]
    return txt_files

def select_file(files):
    print("Select a file to read:")
    for i, file in enumerate(files):
        print(f"{i+1}. {file}")
    
    while True:
        try:
            choice = int(input("Enter the number of the file you want to read: "))
            if choice < 1 or choice > len(files):
                raise ValueError
            break
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    
    return files[choice - 1]

directory = "results/"

txt_files = list_txt_files(directory)

if txt_files:
    selected_file = select_file(txt_files)
else:
    print("No .txt files found in the specified directory.")



# Define a dictionary to store dataset names and their respective accuracies
dataset_accuracies = {}
fp = f'results/{selected_file}'
# Open the text file and read its content
with open(fp, 'r') as file:
    lines = file.readlines()

# Print the header
print(f"{'Dataset Name'.ljust(30)} Accuracy\t\tExecution Time")
print('-'*70)

# Iterate through each line in the file
for line in lines:
    # Use regular expressions to extract dataset name, accuracy, and execution time
    dataset_match = re.search(r'DATASET:\s(.*?)\n', line)
    accuracy_match = re.search(r'Average test multivariate accuracy:\s(\d+\.\d+)\s±\s(\d+\.\d+)', line)
    time_match = re.search(r'Average elapsed time:\s(\d+\.\d+)\s±\s(\d+\.\d+)', line)

    if dataset_match:
        dataset_name = dataset_match.group(1)
    if accuracy_match :
        accuracy = float(accuracy_match.group(1))
        std = float(accuracy_match.group(2))
    if time_match:
        execution_time = float(time_match.group(1))
        # Format the output to include accuracy and execution time in separate columns
        print(f"{dataset_name.ljust(30)} {100*accuracy:.3f} ± {100*std:.3f}\t\t{execution_time:.3f}")

    