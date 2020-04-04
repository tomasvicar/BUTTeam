import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset_class import Dataset
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
import net
from torch import optim
from compute_beta_score import compute_beta_score
from compute_beta_score_tom import compute_beta_score_tom
from shutil import copyfile
from find_best_t import get_best_ts


PATHS = {"labels": "../Partitioning/data/partition/",
         "data": "../../Training_WFDB/",
         }



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_partition_data(file_name, file_path):
    with open(os.path.join(file_path, file_name)) as json_data:
        return json.load(json_data)
    
    
def wce(res,lbls,w_positive_tensor,w_negative_tensor):
    
    res_c = torch.clamp(res,min=1e-6,max=1-1e-6)
            
    p1=lbls*torch.log(res_c)*w_positive_tensor
    p2=(1-lbls)*torch.log(1-res_c)*w_negative_tensor
    
    return -torch.mean(p1+p2)
    

def beta_loss(output, labels,w_positive_tensor,w_negative_tensor,beta=2):

    smooth = 0.1

    TP=(labels)*(output)
    FP=(1-labels)*(output)
    FN=(labels)*(1-output)
    
    
    num_labels=torch.sum(labels,dim=1,keepdims =True)
    
    TP=TP/num_labels
    FP=FP/num_labels
    FN=FN/num_labels
    

    TP=torch.sum(TP,dim=0)/num_labels
    FP=torch.sum(FP,dim=0)/num_labels
    FN=torch.sum(FN,dim=0)/num_labels
    


    
    Fbetas=((1+beta**2)*TP+smooth)/((1+beta**2)*TP+FP+beta**2*FN+smooth)
    
    Gbetas=(TP+smooth)/(TP+FP+beta*FN+smooth)
                        
    Fbeta=torch.mean(Fbetas)
    Gbeta=torch.mean(Gbetas)
    
    
    return -torch.sqrt(Fbeta*Gbeta)



if __name__ == "__main__":
    
    # CUDA for PyTorch
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    device = torch.device("cuda:0")

    # Parameters
    params = {"batch_size": 16,
              "shuffle": True,
              "num_workers":4,
              'collate_fn':Dataset.collate_fn}
    
    max_epochs = 122
    step_size=40
    gamma=0.1
    init_lr=0.01
    save_dir='../../tmp'
    model_note='aug_attentintest_best_t'
    
    best_t=1
    # loss_fcn=beta_loss
    loss_fcn=wce
    
    
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
    training_set = Dataset(partition["train"], PATHS["data"],'train')
    training_generator = data.DataLoader(training_set, **params)

    validation_set = Dataset(partition["validation"], PATHS["data"],'valid')
    validation_generator = data.DataLoader(validation_set, **params)

    # Model import
    model = net.Net_addition_grow()
    
    # model_name='best_models' + os.sep  + 'aug_adition_net_best_t_smalerbatch_larger_model_vicaug55_0.001_train_0.82792526_valid_0.75376415.pkl'
    # model=torch.load(model_name)
    

    model=model.to(device)


    optimizer = optim.Adam(model.parameters(),lr = init_lr ,betas= (0.9, 0.999),eps=1e-8,weight_decay=1e-8)
    scheduler=optim.lr_scheduler.StepLR(optimizer, step_size, gamma=gamma, last_epoch=-1)
    
    
    
    trainig_loss_log=[]
    trainig_beta_log=[]
    
    valid_loss_log=[]
    valid_beta_log=[]
    
    model_names=[]
    
    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        
        lbls_np_log=[]
        res_np_log=[]
        tmp_loss_log=[]
        
        model.train()
        
        for pad_seqs,lens,lbls in training_generator:
            
            pad_seqs_np,lens_np,lbls_np = pad_seqs.detach().cpu().numpy(),lens.detach().cpu().numpy(),lbls.detach().cpu().numpy()
            # Transfer to GPU
            # pad_seqs,lens,lbls = pad_seqs.to(device),lens.to(device),lbls.to(device)
            pad_seqs,lens,lbls = pad_seqs.cuda(0),lens.cuda(0),lbls.cuda(0)
            
            
            
            res=model(pad_seqs,lens)
            
            res_np = res.detach().cpu().numpy()
            
            
            
            w_positive_tensor=torch.from_numpy(w_positive.astype(np.float32)).cuda(0)
            w_negative_tensor=torch.from_numpy(w_negative.astype(np.float32)).cuda(0)
            
            

            
            loss=loss_fcn(res,lbls,w_positive_tensor,w_negative_tensor)

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_np=loss.detach().cpu().numpy()

            
            lbls_np_log.append(lbls_np)
            res_np_log.append(res_np)
            tmp_loss_log.append(loss_np)
      
            
            
            
            
            
        lbls_np_log=np.concatenate(lbls_np_log,axis=0)
        res_np_log=np.concatenate(res_np_log,axis=0)
            
        if best_t:
            t=get_best_ts(res_np_log,lbls_np_log)
        else:
            t=0.5
            
            
        Fbeta,Gbeta,geom_mean= compute_beta_score_tom(lbls_np_log, res_np_log>t, 2, 9)
        
        
        
        trainig_beta_log.append(geom_mean)
        trainig_loss_log.append(np.mean(tmp_loss_log))

            
            
            
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
            
            

            loss=loss_fcn(res,lbls,w_positive_tensor,w_negative_tensor)
            
            

            
            loss_np=loss.detach().cpu().numpy()

            
            lbls_np_log.append(lbls_np)
            res_np_log.append(res_np)
            tmp_loss_log.append(loss_np)
            
            
            
            
            
            
            
            
            
            
            
        lbls_np_log=np.concatenate(lbls_np_log,axis=0)
        res_np_log=np.concatenate(res_np_log,axis=0)
          
        if best_t:
            t=get_best_ts(res_np_log,lbls_np_log)
            model.set_t(t[0,:])
        else:
            t=0.5
        Fbeta,Gbeta,geom_mean= compute_beta_score_tom(lbls_np_log, res_np_log>t, 2, 9)
    
        
        valid_beta_log.append(geom_mean)
        valid_loss_log.append(np.mean(tmp_loss_log))
   
        
        
        plt.plot(trainig_loss_log,'b')
        plt.plot(valid_loss_log,'r')
        plt.title('loss')
        plt.show()
        
        
        plt.plot(trainig_beta_log,'b')
        plt.plot(valid_beta_log,'g')
        plt.title('geometric mean')
        plt.show()
        
        lr=get_lr(optimizer)
        
        info=str(epoch) + '_' + str(lr) + '_train_'  + str(trainig_beta_log[-1]) + '_valid_' + str(valid_beta_log[-1]) 
        
        model_name=save_dir+ os.sep + model_note +info  + '.pkl'
        torch.save(model,model_name)
            
        model_names.append(model_name)
        
        print(info)
        
        
        
        scheduler.step()
            
          
                
    best_models_dir='best_models' 
        
    best_model_ind=np.argmax(valid_beta_log)
    best_model_name= model_names[best_model_ind]   
    best_model_name_new=best_model_name.replace(save_dir,best_models_dir)   
            
            
    try:
        os.mkdir(best_models_dir)
    except:
        pass            
    
    copyfile(best_model_name, best_model_name_new)
                













