import os
from shutil import copyfile,rmtree
import numpy as np


DATA_PATH= "../../Training_WFDB"

path_test= "../../test_split_tmp"
path_train= "../../train_split_tmp"

split_ratio=[8,2]


try:
    rmtree(path_test)
except:
    pass

try:
    rmtree(path_train)
except:
    pass



try:
    os.mkdir(path_test)
except:
    pass

try:
    os.mkdir(path_train)
except:
    pass




names=[]
for root, dirs, files in os.walk(DATA_PATH):
    for name in files:
        if name.endswith(".mat"):
            name=name.replace('.mat','')
            names.append(name)


split_ratio_ind=int(np.floor(split_ratio[0]/(split_ratio[0]+split_ratio[1])*len(names)))
perm=np.random.permutation(len(names))
train_ind=perm[:split_ratio_ind]
valid_ind=perm[split_ratio_ind:]


for k in train_ind:
    name_in=DATA_PATH + os.sep + names[k]
    name_out=path_train + os.sep + names[k]
    copyfile(name_in + '.mat',name_out+'.mat')
    copyfile(name_in + '.hea',name_out+'.hea')
  

for k in valid_ind:
    name_in=DATA_PATH + os.sep + names[k]
    name_out=path_test + os.sep + names[k]
    copyfile(name_in + '.mat',name_out+'.mat')
    copyfile(name_in + '.hea',name_out+'.hea')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    