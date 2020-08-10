import numpy as np
from utils.compute_challenge_metric_custom import compute_challenge_metric_custom
from bayes_opt import BayesianOptimization
from config import Config



def aply_ts(res_all,ts):

    res_binar=np.zeros(res_all.shape,dtype=np.bool)

    for class_num in range(len(ts)):

        res_binar[:,class_num]=res_all[:,class_num]>ts['t' + str(class_num)]

    return res_binar


def optimize_ts(res_all,lbls_all,fast=False):
    
    def evaluate_ts(normalize=False,**ts):
        
        res_binar=aply_ts(res_all,ts)
            
        challenge_metric=compute_challenge_metric_custom(res_binar,lbls_all,normalize=normalize)
        
        return challenge_metric
    
    
    func = evaluate_ts  

    param_names=['t' + str(k) for k in range(lbls_all.shape[1])]
    bounds_lw=0*np.ones(lbls_all.shape[1])
    bounds_up=1*np.ones(lbls_all.shape[1])
    
    
    pbounds=dict(zip(param_names, zip(bounds_lw,bounds_up)))
    
    optimizer = BayesianOptimization(f=func,pbounds=pbounds,random_state=1)  
      
    if fast:
        optimizer.maximize(init_points=200,n_iter=0)
    else:
        optimizer.maximize(init_points=Config.T_OPTIMIZE_INIT,n_iter=Config.T_OPTIMIZER_GP)
    
    
    ts=optimizer.max['params']
    
    
    res_binar=aply_ts(res_all,ts)
            
    opt_challenge_metric=compute_challenge_metric_custom(res_binar,lbls_all,normalize=True)
    
    
    
    return ts,opt_challenge_metric