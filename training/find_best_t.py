import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset_class import Dataset
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
from net import Net
from torch import optim
from compute_beta_score import compute_beta_score
from compute_beta_score_tom import compute_beta_score_tom
from shutil import copyfile
import time



def get_best_ts(res_np_log,lbls_np_log):

    res_np_log
    t=np.zeros(np.shape(res_np_log))
    for f_num in range(res_np_log.shape[1]):
        
        
        f=res_np_log[:,[f_num]]
        l=lbls_np_log[:,[f_num]]
        
        t_best=0.5
        v_best=-1
        for tt in np.concatenate( (np.linspace(0,0.1,100),np.linspace(0,1,100),np.linspace(0,0.1,100)),axis=0):
            
            Fbeta,Gbeta,v= compute_beta_score_tom(l, f>tt, 2, 1)
            if v>v_best:
                v_best=v
                t_best=tt
            
        
        
        t[:,f_num]=t_best

    return t







if __name__ == "__main__":
    
    PATHS = {"labels": "../Partitioning/data/partition/",
         "data": "../../Training_WFDB/",
         }

    
    
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    
    def get_partition_data(file_name, file_path):
        with open(os.path.join(file_path, file_name)) as json_data:
            return json.load(json_data)
    
    # CUDA for PyTorch
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    device = torch.device("cuda:0")

    # Parameters
    params = {"batch_size": 1,
              "shuffle": False,
              "num_workers": 0,
              "drop_last": False,
              'collate_fn':Dataset.collate_fn}
    
    max_epochs = 62
    step_size=20
    gamma=0.1
    init_lr=0.01
    save_dir='../../tmp'
    
    try:
        os.mkdir(save_dir)
    except:
        pass
    
    
    
    
    lbl_counts=np.array([ 728,  955,  582,  189, 1448,  487,  550,  691,  179])
    num_of_sigs=5430
    w_positive=num_of_sigs/lbl_counts
    w_negative=num_of_sigs/(num_of_sigs-lbl_counts)
    
    
    

    # Datasets
    partition = get_partition_data("partition_82.json", PATHS["labels"])

    # Generators

    validation_set = Dataset(partition["validation"], PATHS["data"])
    validation_generator = data.DataLoader(validation_set, **params)



    # Model import
    # model = Net()
    model_name='best_models' + os.sep  + '61_1e-05_train_0.9286569_valid_0.8222659.pkl'
    model=torch.load(model_name)
    
    # model=model.cuda(0)
    model=model.to(device)



    
    valid_loss_log=[]
    valid_beta_log=[]
        
        
            
    lbls_np_log=[]
    res_np_log=[] 
    
    model.eval()
        
    for pad_seqs,lens,lbls in validation_generator:
        
        pad_seqs_np,lens_np,lbls_np = pad_seqs.detach().cpu().numpy(),lens.detach().cpu().numpy(),lbls.detach().cpu().numpy()
        # Transfer to GPU
        # pad_seqs,lens,lbls = pad_seqs.to(device),lens.to(device),lbls.to(device)
        pad_seqs,lens,lbls = pad_seqs.cuda(0),lens.cuda(0),lbls.cuda(0)
        
        
        
        res=model(pad_seqs,lens)
        
        res_np = res.detach().cpu().numpy()
        
        
        
        lbls_np_log.append(lbls_np)
        res_np_log.append(res_np)
        
        

        
        
        
    lbls_np_log=np.concatenate(lbls_np_log,axis=0)
    res_np_log=np.concatenate(res_np_log,axis=0)
        
    Fbeta,Gbeta,geom_mean05= compute_beta_score_tom(lbls_np_log, res_np_log>0.5, 2, 9)

    

    t=get_best_ts(res_np_log,lbls_np_log)
    
    

    Fbeta,Gbeta,geom_mean_best= compute_beta_score_tom(lbls_np_log, res_np_log>t, 2, 9)
    
    
    _,_,Fbeta_measure_best2,Gbet_measure_best2= compute_beta_score(lbls_np_log, res_np_log>t, 2, 9)
    
    geom_mean_best2=np.sqrt(Fbeta_measure_best2*Gbet_measure_best2)

    print(geom_mean_best)
    print(geom_mean_best2)
    print(geom_mean05)


    print(t[0,:])




