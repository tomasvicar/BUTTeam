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


PATHS = {"labels": "../Partitioning/data/partition/",
         "data": "../../Training_WFDB/",
         }



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_partition_data(file_name, file_path):
    with open(os.path.join(file_path, file_name)) as json_data:
        return json.load(json_data)


if __name__ == "__main__":
    
    # CUDA for PyTorch
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    device = torch.device("cuda:0")

    # Parameters
    params = {"batch_size": 64,
              "shuffle": True,
              "num_workers": 2,
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
    training_set = Dataset(partition["train"], PATHS["data"])
    training_generator = data.DataLoader(training_set, **params)

    validation_set = Dataset(partition["validation"], PATHS["data"])
    validation_generator = data.DataLoader(validation_set, **params)

    # Model import
    # model = Net()
    model_name='best_models' + os.sep  + '61_1e-05_train_0.9286569_valid_0.8222659.pkl'
    model=torch.load(model_name)
    
    # model=model.cuda(0)
    model=model.to(device)


    optimizer = optim.Adam(model.parameters(),lr = init_lr ,betas= (0.9, 0.999),eps=1e-8,weight_decay=1e-8)
    scheduler=optim.lr_scheduler.StepLR(optimizer, step_size, gamma=gamma, last_epoch=-1)
    
    
    
    trainig_loss_log=[]
    trainig_beta_log=[]
    
    valid_loss_log=[]
    valid_beta_log=[]
    
    model_names=[]
        
        
            
    lbls_np_log=[]
    res_np_log=[] 
    tmp_loss_log=[]
    
    model.eval()
        
    for pad_seqs,lens,lbls in validation_generator:
        
        pad_seqs_np,lens_np,lbls_np = pad_seqs.detach().cpu().numpy(),lens.detach().cpu().numpy(),lbls.detach().cpu().numpy()
        # Transfer to GPU
        # pad_seqs,lens,lbls = pad_seqs.to(device),lens.to(device),lbls.to(device)
        pad_seqs,lens,lbls = pad_seqs.cuda(0),lens.cuda(0),lbls.cuda(0)
        
        
        
        res=model(pad_seqs,lens)
        
        res_np = res.detach().cpu().numpy()
        
        
        
        w_positive_tensor=torch.from_numpy(w_positive.astype(np.float32)).cuda(0)
        w_negative_tensor=torch.from_numpy(w_negative.astype(np.float32)).cuda(0)
        
        

        
        res_c = torch.clamp(res,min=1e-6,max=1-1e-6)
        
        res_c_np=res_c.detach().cpu().numpy()
        
        p1=lbls*torch.log(res_c)*w_positive_tensor
        p2=(1-lbls)*torch.log(1-res_c)*w_negative_tensor
        p1_np=p1.detach().cpu().numpy()
        p2_np=p2.detach().cpu().numpy()
        loss=-torch.mean(p1+p2)
        # loss=F.binary_cross_entropy(res,lbls)
        
        

        
        loss_np=loss.detach().cpu().numpy()

        
        lbls_np_log.append(lbls_np)
        res_np_log.append(res_np)
        tmp_loss_log.append(loss_np)
        
        

        
        
        
    lbls_np_log=np.concatenate(lbls_np_log,axis=0)
    res_np_log=np.concatenate(res_np_log,axis=0)
        
    Fbeta_measure05= compute_beta_score_tom(lbls_np_log, res_np_log>0.5, 2, 9)
    
    
    time1 = time.time()
    res_np_log
    t=np.zeros(np.shape(res_np_log))
    for f_num in range(res_np_log.shape[1]):
        
        
        f=res_np_log[:,[f_num]]
        l=lbls_np_log[:,[f_num]]
        
        t_best=0.5
        v_best=-1
        for t in np.linspace(0,1,1000):
            t
            
            v= compute_beta_score_tom(l, f>t, 2, 1)
            if v>=v_best:
                v_best=v
                best_t=t
            
        
        
        t[:,f_num]=best_t

        
    print(time.time()-time1)    
        

        
    time1 = time.time()
    Fbeta_measure_best= compute_beta_score_tom(lbls_np_log, res_np_log>t, 2, 9)
    print(time.time()-time1)
    

    print(Fbeta_measure_best)
    print(Fbeta_measure05)







