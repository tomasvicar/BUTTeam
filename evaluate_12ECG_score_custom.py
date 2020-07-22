import numpy as np
from evaluate_12ECG_score import compute_challenge_metric,load_weights
from compute_challenge_metric_custom import compute_challenge_metric_custom


def evaluate_12ECG_score_custom(res,lbls,classes):
    
    
    weights_file = 'weights.csv'
    normal_class = '426783006'
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]
    
    
    labels=lbls
    binary_outputs=res
    
    weights = load_weights(weights_file, classes)
    
    indices = np.any(weights, axis=0) # Find indices of classes in weight matrix.
    classes = [x for i, x in enumerate(classes) if indices[i]]
    labels = labels[:, indices]
    binary_outputs = binary_outputs[:, indices]
    weights = weights[np.ix_(indices, indices)]
    
    
    challenge_metric = compute_challenge_metric(weights, labels, binary_outputs, classes, normal_class)
    
    
    challenge_metric = compute_challenge_metric(weights, labels, binary_outputs, classes, normal_class)