import numpy as np
import glob
import matplotlib.pyplot as plt

from utils.get_stats import get_stats,get_stats2

input_directory = '../data_nofold'

file_list = glob.glob(input_directory + "/**/*.mat", recursive=True)

num_files = len(file_list)
print(num_files)

permuted_idx = np.random.permutation(num_files)
file_list=[file_list[file_idx] for file_idx in permuted_idx[:2000]]

lbl_counts,lens,stds,res=get_stats2(file_list)



plt.hist(stds[:,0],50)
plt.show()

file_list2=[file_list[file_idx] for file_idx in np.argsort(stds[:,0])[-50:]]

res[np.argsort(stds[:,0])[:500]]





