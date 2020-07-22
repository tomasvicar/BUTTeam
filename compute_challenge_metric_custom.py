import numpy as np
from config import Config



def compute_challenge_metric_custom(res,lbls):
    
    
    normal_class = '426783006'
    normal_index=Config.HASH_TABLE['snomeds'].index(normal_class)

    lbls=lbls>0
    res=res>0
    
    weights = Config.loaded_weigths
    
    observed_score=np.sum(weights*get_confusion(lbls,res))
    
    correct_score=np.sum(weights*get_confusion(lbls,lbls))
    
    inactive_outputs = np.zeros_like(lbls)
    inactive_outputs[:, normal_index] = 1
    inactive_score=np.sum(weights*get_confusion(lbls,inactive_outputs))
    
    normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
    
    return normalized_score


        
def get_confusion(lbls,res):

    

    normalizer=np.sum(lbls|res,axis=1)
    num_sigs,num_classes=lbls.shape
    
    A=np.zeros((num_classes,num_classes))
    for sig_num in range(num_sigs):
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if lbls[sig_num, j]:
                for k in range(num_classes):
                    if res[sig_num, k]:
                        A[j, k] += 1.0/normalizer[sig_num]
    
    
    return A
        