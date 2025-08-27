'''Function responsible for logging the training process.'''
import numpy as np


def save_histories_to_file(configs, histories, output_path_summary, output_path_full, prefix=""):
    string_summ = prefix + "\n"
    string_full = prefix + "\n"
    for config, history in zip(configs, histories):
        elapsed_times, N, multiv_info, train_leaf_info, test_leaf_info = zip(*history)
        multiv_acc_in, multiv_acc_test = zip(*multiv_info)
        train_active_leaves, train_normalized_entropy = zip(*train_leaf_info)
        test_active_leaves, test_normalized_entropy = zip(*test_leaf_info)

        

        string_summ += "--------------------------------------------------\n\n"
        string_summ += f"DATASET: {config['name']}\n"
        string_summ += f"Num Total Samples: {N[0]}\n"
        string_summ += f"{len(elapsed_times)} simulations executed. Seed: {config['tseed']}\n"
        string_summ += f"Average in-sample multivariate accuracy: {'{:.3f}'.format(np.mean(multiv_acc_in))} ± {'{:.3f}'.format(np.std(multiv_acc_in))}\n"
        string_summ += f"Average test multivariate accuracy: {'{:.3f}'.format(np.mean(multiv_acc_test))} ± {'{:.3f}'.format(np.std(multiv_acc_test))}\n"
        string_summ += f"Average train active leaves: {'{:.3f}'.format(np.mean(train_active_leaves))} ± {'{:.3f}'.format(np.std(train_active_leaves))}\n"
        string_summ += f"Average train normalized entropy: {'{:.3f}'.format(np.mean(train_normalized_entropy))} ± {'{:.3f}'.format(np.std(train_normalized_entropy))}\n"
        string_summ += f"Average test active leaves: {'{:.3f}'.format(np.mean(test_active_leaves))} ± {'{:.3f}'.format(np.std(test_active_leaves))}\n"
        string_summ += f"Average test normalized entropy: {'{:.3f}'.format(np.mean(test_normalized_entropy))} ± {'{:.3f}'.format(np.std(test_normalized_entropy))}\n"
        string_summ += "\n"
        string_summ += f"Best test multivariate accuracy: {'{:.3f}'.format(multiv_acc_test[np.argmin(multiv_acc_test)])}\n"
        string_summ += f"Average elapsed time: {'{:.3f}'.format(np.mean(elapsed_times))} ± {'{:.3f}'.format(np.std(elapsed_times))}\n"

        string_full += "--------------------------------------------------\n\n"
        string_full += f"DATASET: {config['name']}\n"

        for (elapsed_time, \
             (N),
             (multiv_acc_in, multiv_acc_test),
             (train_active_leaves, train_normalized_entropy), 
             (test_active_leaves, test_normalized_entropy)
             ) in history:
            string_full += f"In-sample:" + "\n"
            string_full += f"        Multivariate accuracy: {multiv_acc_in}" + "\n"
            string_full += f"Test:" + "\n"
            string_full += f"        Multivariate accuracy: {multiv_acc_test}" + "\n"
            string_full += f"        Train Active Leaves: {train_active_leaves}" + "\n"
            string_full += f"        Train Normalized Entropy: {train_normalized_entropy:.2f}" + "\n"
            string_full += f"        Test Active Leaves: {test_active_leaves}" + "\n"
            string_full += f"        Test Normalized Entropy: {test_normalized_entropy:.2f}" + "\n"
            string_full += f"Elapsed time: {elapsed_time}" + "\n"
            
            string_full += "\n\n--------\n\n"

    with open(output_path_summary, "w", encoding="utf-8") as text_file:
        text_file.write(string_summ)

    with open(output_path_full, "w", encoding="utf-8") as text_file:
        text_file.write(string_full)
