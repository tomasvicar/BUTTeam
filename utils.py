import torch
import numpy as np

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    
    
def wce(res,lbls,w_positive_tensor,w_negative_tensor):
    ## weighted crossetropy - weigths are for positive and negative 
    res_c = torch.clamp(res,min=1e-6,max=1-1e-6)
            
    p1=lbls*torch.log(res_c)*w_positive_tensor
    p2=(1-lbls)*torch.log(1-res_c)*w_negative_tensor
    
    return -torch.mean(p1+p2)


def snomed2hot(snomed,HASH_TABLE):
    y=np.zeros((len(HASH_TABLE),1)).astype(np.float32)
    for kk,p in enumerate(HASH_TABLE):
        for lbl_i in snomed:
            if lbl_i.find(p)>-1:
                y[kk]=1
                
    return y