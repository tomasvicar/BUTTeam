import numpy as np

def get_dice(lbls_all,res_all):
    TP=np.sum((lbls_all==1)&(res_all==1),axis=0)
    FP=np.sum((lbls_all==0)&(res_all==1),axis=0)
    FN=np.sum((lbls_all==1)&(res_all==0),axis=0)
    TN=np.sum((lbls_all==0)&(res_all==0),axis=0)
    
    dice = (2*TP)/(2*TP + FP + FN)
    
    return np.mean(dice)
