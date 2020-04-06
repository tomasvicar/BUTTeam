import os
from read_file import read_data
from read_file import read_lbl
import numpy as np
from config import Config


def get_splits_means_stds_lens_counts():

    
    DATA_PATH=Config.DATA_PATH
    split_ratio=Config.split_ratio
    num_of_splits=Config.num_of_splits
    
    
    pato_names=Config.pato_names
    
    try:
        os.mkdir(Config.info_save_dir)
    except:
        pass
    
    
    
    names=[]
    for root, dirs, files in os.walk(DATA_PATH):
        for name in files:
            if name.endswith(".mat"):
                name=name.replace('.mat','')
                names.append(name)
    
    
    labels=[]
    means=[]
    stds=[]
    lens=[]
    for k,file_name in enumerate(names):
    
        
        X = read_data(DATA_PATH, file_name)
        
        means.append(np.mean(X,axis=1))
        stds.append(np.std(X,axis=1))
        lens.append(X.shape[1])
        
        
        lbl = read_lbl(DATA_PATH, file_name)
        
        
        labels.append(lbl)
        
        
    MEANS=np.mean(np.stack(means,axis=1),axis=1)
    STDS=np.mean(np.stack(stds,axis=1),axis=1)
    
    
    
    
    
    more_hot_lbls=[]
    for k,lbl in enumerate(labels):
          
        res=np.zeros(len(pato_names))
        
        lbl=lbl.split(',')
    
        for kk,p in enumerate(pato_names):
            for lbl_i in lbl:
                if lbl_i.find(p)>-1:
                    res[kk]=1
                
        more_hot_lbls.append(res>0)
        
        
    
    tmp=np.stack(more_hot_lbls,axis=1)
    
    lbl_counts=np.sum(tmp,axis=1)
    
    num_of_sigs=len(lens)
    
    
    
    print(MEANS)
    
    
    print(STDS)
    
    
    print(lbl_counts)
    
    
    print(len(lens))
    
    
    
    np.save(Config.info_save_dir+os.sep+'MEANS.npy', np.array(MEANS))
    np.save(Config.info_save_dir+os.sep+'STDS.npy', np.array(STDS))
    np.save(Config.info_save_dir+os.sep+'lbl_counts.npy', np.array(lbl_counts))
    np.save(Config.info_save_dir+os.sep+'lens.npy', np.array(lens))
    
    
    
    np.random.seed(666)
    
    splits=[]
    for k in range(num_of_splits):
        split_ratio_ind=int(np.floor(split_ratio[0]/(split_ratio[0]+split_ratio[1])*len(names)))
        perm=np.random.permutation(len(names))
        train_ind=perm[:split_ratio_ind]
        valid_ind=perm[split_ratio_ind:]
        split= {'train': [names[i] for i in train_ind],'valid': [names[i] for i in valid_ind]}
        splits.append(split)
    
    
    
    np.save(Config.info_save_dir+os.sep+'splits.npy', splits)
    
    np.save(Config.info_save_dir+os.sep+'num_of_sigs.npy', num_of_sigs)


