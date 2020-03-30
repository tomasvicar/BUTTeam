import numpy as np


def compute_beta_score_tom(labels, output, beta=2, num_classes=9):
    
    
  
    TP=(labels==1)&(output==1)
    FP=(labels==0)&(output==1)
    FN=(labels==1)&(output==0)
    
    num_labels=np.sum(labels,axis=1,keepdims =1)
    
    
    TP=TP/num_labels
    FP=FP/num_labels
    FN=FN/num_labels
    

    TP=np.sum(TP,axis=0)/num_labels
    FP=np.sum(FP,axis=0)/num_labels
    FN=np.sum(FN,axis=0)/num_labels
    


    
    Fbetas=(1+beta**2)*TP/((1+beta**2)*TP+FP+beta**2*FN)
    
    
    Fbetas[np.isnan(Fbetas)]=1
    
    
    Fbeta=np.mean(Fbetas)
    
    
    
    
    return Fbeta
    
    
    
    
    
    
    



