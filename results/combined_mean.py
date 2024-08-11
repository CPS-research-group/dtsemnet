import numpy as np


## CTSlice DTSemNet
# 1.743 ± 0.134
# 1.799 ± 0.246
# 1.834 ± 0.214
# 1.804 ± 0.158
# 1.855 ± 0.160

## CTSlice2 DTSemNet
# 1.430 ± 0.105
# 1.431 ± 0.134
# 1.465 ± 0.139
# 1.442 ± 0.104
# 1.470 ± 0.115

## ctlice DGT
# 1.782 ± 0.252
# 1.875 ± 0.457
# 1.686 ± 0.116
# 1.819 ± 0.082
# 1.739 ± 0.086

########

## abalone DTSemNet
# 2.115 ± 0.036
# 2.130 ± 0.010
# 2.186 ± 0.022
# 2.129 ± 0.006
# 2.119 ± 0.021

# abalone DGT
# 2.139 ± 0.038
# 2.133 ± 0.008
# 2.178 ± 0.019
# 2.156 ± 0.015
# 2.116 ± 0.013


#########
## cpu active DGT
# 2.765 ± 0.116
# 2.618 ± 0.072
# 2.544 ± 0.072
# 2.691 ± 0.225
# 2.606 ± 0.111

## cpu active dtsement
# 2.763 ± 0.052
# 2.605 ± 0.058
# 2.629 ± 0.109
# 2.717 ± 0.129
# 2.705 ± 0.127

## cpu active dtsement2
# 2.699 ± 0.072
# 2.602 ± 0.084
# 2.610 ± 0.061
# 2.734 ± 0.365
# 2.581 ± 0.021

######
# mu = np.array([1.743, 1.799, 1.834, 1.804, 1.855])
# std = np.array([0.134, 0.246, 0.214, 0.158, 0.160])
# mu = np.array([2.115, 2.130, 2.186, 2.129, 2.119])
# std = np.array([0.036, 0.010, 0.022, 0.006, 0.021])
# mu = np.array([1.430, 1.431, 1.465, 1.442, 1.470])
# std = np.array([0.105, 0.134, 0.139, 0.104, 0.115])
# mu = np.array([2.139, 2.133, 2.178, 2.156, 2.116])
# std = np.array([0.038, 0.008, 0.019, 0.015, 0.013])

# mu = np.array([1.782, 1.875, 1.686, 1.819, 1.739])
# std = np.array([0.252, 0.457, 0.116, 0.082, 0.086])

mu = np.array([2.765, 2.618, 2.544, 2.691, 2.606])
std = np.array([0.116, 0.072, 0.072, 0.225, 0.111])
# mu = np.array([2.763, 2.605, 2.629, 2.717, 2.705])
# std = np.array([0.052, 0.058, 0.109, 0.129, 0.127])

# mu = np.array([2.699, 2.602, 2.610, 2.734, 2.581])
# std = np.array([0.072, 0.084, 0.061, 0.365, 0.021])

combined_mean = np.mean(mu)
print(combined_mean)
diff_mean = mu - combined_mean

all_std = np.concatenate((std, diff_mean), axis=None)
print(all_std)
combined_std = np.sqrt(np.sum(np.square(all_std)) / len(std))

print(f"Combined mean: {combined_mean}")
print(f"Combined standard deviation: {combined_std}")