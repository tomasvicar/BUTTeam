import os
import json
from read_file import read_data
from read_file import read_lbl_tom
import numpy as np


PATHS = {"labels": "../Partitioning/data/partition/",
         "data": "../../Training_WFDB/",
         }


def get_partition_data(file_name, file_path):
    with open(os.path.join(file_path, file_name)) as json_data:
        return json.load(json_data)
    


partition = get_partition_data("partition_82.json", PATHS["labels"])
#labels = get_partition_data("labels.json", PATHS["labels"])

# names=partition["train"]

# names=partition["train"]+partition["validation"]

names=partition["validation"]

#labels_train =[labels[x] for x in partition["train"]]


labels_train=[]
means=[]
stds=[]
lens=[]
for k,file_name in enumerate(names):

    
    X = read_data(PATHS["data"], file_name)
    
    means.append(np.mean(X,axis=1))
    stds.append(np.std(X,axis=1))
    lens.append(X.shape[1])
    
    
    lbl = read_lbl_tom(PATHS["data"], file_name)
    
    
    labels_train.append(lbl)
    
    
MEANS=np.mean(np.stack(means,axis=1),axis=1)
STDS=np.mean(np.stack(stds,axis=1),axis=1)


pato_names=['Normal','AF','I-AVB','LBBB','RBBB','PAC','PVC','STD','STE']


more_hot_lbls=[]
for k,lbl in enumerate(labels_train):
      
    res=np.zeros(len(pato_names))
    
    lbl=lbl.split(',')

    for kk,p in enumerate(pato_names):
        for lbl_i in lbl:
            if lbl_i.find(p)>-1:
                res[kk]=1
            
    more_hot_lbls.append(res>0)
    
    

tmp=np.stack(more_hot_lbls,axis=1)

lbl_counts=np.sum(tmp,axis=1)



print(np.sort(lens)[:10])
# [3000 4000 4500 4500 4500 4500 4999 4999 4999 4999]

print(np.sort(lens)[-10:])
# [37000 42500 44000 48500 49000 50500 56000 59000 66000 69000]

print(MEANS)
# [-1.51309233e-03 -4.87992638e-04  4.42326016e-03 -1.16812856e-03
#  -8.32427227e-04 -7.69686243e-06  4.24138563e-04 -2.47998114e-03
#  -2.50843292e-03 -1.22554633e-03  7.08855026e-04  3.02959096e-03]

print(STDS)
# [121.40858639 149.55139422 121.14471528 124.44668018  96.85791404
#  120.87596136 204.83819888 295.70214234 300.9895724  309.04986076
#  291.26254274 260.78131754]

print(lbl_counts)
# [ 728  955  582  189 1448  487  550  691  179]


print(len(lens))
# 5430


np.save('lens.npy', np.array(lens))

