import torch
import numpy as np    
from utils.utils import load_weights
from utils.datareader import DataReader

def wce(res,lbls,w_positive_tensor,w_negative_tensor):
    ## weighted crossetropy - weigths are for positive and negative 
    res_c = torch.clamp(res,min=1e-6,max=1-1e-6)
            
    p1=lbls*torch.log(res_c)*w_positive_tensor
    p2=(1-lbls)*torch.log(1-res_c)*w_negative_tensor
    
    return -torch.mean(p1+p2)

def challange_metric_loss(res,lbls,w_positive_tensor,w_negative_tensor):
    ## weighted crossetropy - weigths are for positive and negative 
    
    
    
    normalizer=torch.sum(lbls+res-lbls*res,dim=1)
    normalizer[normalizer<1]=1
    
    num_sigs,num_classes=list(lbls.size())
    
    
    weights = torch.from_numpy(load_weights('weights.csv',list(DataReader.get_label_maps(path="tables/")[0].keys())).astype(np.float32))
    A=torch.zeros((num_classes,num_classes),dtype=lbls.dtype)
    
    cuda_check = lbls.is_cuda
    if cuda_check:
        cuda_device = lbls.get_device()
        device = torch.device('cuda:' + str(cuda_device) )
        A=A.to(device)
        weights=weights.to(device)


    for sig_num in range(num_sigs):
        tmp=torch.matmul(torch.transpose(lbls[[sig_num], :],0,1),res[[sig_num], :])/normalizer[sig_num]
        A=A + tmp
    
    return -torch.sum(A*weights)



class FocalLoss():
    def __init__(self, gamma = 2.0,weighted=False):
        self.gamma=gamma
        self.weighted=weighted
    

    def __call__(self,res,lbls,w_positive_tensor,w_negative_tensor):
        
        gamma=self.gamma
        
        p = torch.clamp(res,min=1e-6,max=1-1e-6)
    
        q=1-p
        
        if self.weighted:
            pos_loss = -(q ** gamma) * torch.log(p)*w_positive_tensor
            neg_loss = -(p ** gamma) * torch.log(q)*w_negative_tensor
        else:
            pos_loss = -(q ** gamma) * torch.log(p)
            neg_loss = -(p ** gamma) * torch.log(q)
    
        loss = lbls * pos_loss + (1 - lbls) * neg_loss
        return torch.mean(loss)




