
import os
import glob
import numpy as np
import glob
import numpy as np
from torch import optim
from torch.utils import data as dataa
import torch
from shutil import copyfile,rmtree
from datetime import datetime



input_directory = '../data_nofold_resfix'


#####################
## modified part

try:
    rmtree(input_directory)
except:
    pass

if not os.path.isdir(input_directory):
    os.mkdir(input_directory)


file_list = glob.glob('../data_resfix' + r"/**/*.mat", recursive=True)
# file_list =[x for x in file_list if 'Training_StPetersburg' not in x]

num_files = len(file_list)
   

for file_num,file in enumerate(file_list):
    path,file_name=os.path.split(file)
    
    copyfile(file,input_directory + os.sep + file_name)
    copyfile(file.replace('.mat','.hea'),input_directory + os.sep + file_name.replace('.mat','.hea'))

