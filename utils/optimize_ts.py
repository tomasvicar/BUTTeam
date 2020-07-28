import numpy as np
from utils.compute_challenge_metric_custom import compute_challenge_metric_custom
from bayes_opt import BayesianOptimization
from config import Config


def aply_ts(scores, ts):
    """
    Applies thresholding to output scores and merge results if sample is divided into batch
    :param scores: model output scores, shape=(batch, 1, num_of_classes)
    :param ts: custom thresholds, shape=(1, num_of_classes)??
    :return: aggregated one hot encoded labels
    """

    def merge_labels(labels):
        """
        Merges labels across single batch
        :param labels: one hot encoded labels, shape=(batch, 1, num_of_classes)
        :return: aggregated one hot encoded labels, shape=(1, 1, num_of_classes)
        """
        return np.amax(labels, axis=0)

    binary_results = np.zeros(scores.shape, dtype=np.bool)
    for class_idx in range(len(ts)):
        binary_results[:, 0, class_idx] = scores[:, 0, class_idx] > ts("t"+str(class_idx))

    return merge_labels(binary_results)


# def aply_ts(res_all,ts):
#
#     res_binar=np.zeros(res_all.shape,dtype=np.bool)
#
#     for class_num in range(len(ts)):
#
#         res_binar[:,class_num]=res_all[:,class_num]>ts['t' + str(class_num)]
#
#     return res_binar


def optimize_ts(res_all,lbls_all):
    
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
      
    optimizer.maximize(init_points=Config.T_OPTIMIZE_INIT,n_iter=Config.T_OPTIMIZER_GP)
    
    
    ts=optimizer.max['params']
    
    
    res_binar=aply_ts(res_all,ts)
            
    opt_challenge_metric=compute_challenge_metric_custom(res_binar,lbls_all,normalize=True)
    
    
    
    return ts,opt_challenge_metric