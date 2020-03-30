#!/usr/bin/env python

import torch
import numpy as np
import os


def run_12ECG_classifier(data,header_data,classes,model):
    
    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)

    # t=model.get_t()
    # t=np.array([0.47847848, 0.42042042, 0.45745746, 0.38838839, 0.15715716, 0.2982983 , 0.49449449, 0.42042042, 0.57757758])
    t=0.5
    
    MEANS=np.array([ 0.00313717,  0.00086543, -0.00454349, -0.00416486,  0.00102769,-0.00275855, -0.00108178,  0.00016227,  0.00010818, -0.00270446,0.00010818, -0.00156859])
    
    STDS=np.array([121.40858639, 149.55139422, 121.14471528, 124.44668018,96.85791404, 120.87596136, 204.83819888, 295.70214234,300.9895724 , 309.04986076, 291.26254274, 260.78131754])
            
    pato_names=['Normal','AF','I-AVB','LBBB','RBBB','PAC','PVC','STD','STE']
    
    order=[];
    for cl in classes:
        for k,pato in enumerate(pato_names):
            if pato==cl:
                order.append(k)
    

    data=(data-MEANS.reshape(-1,1))/STDS.reshape(-1,1)
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    lens=data.shape[1]
    lens=torch.from_numpy(np.array(lens).astype(np.float32)).view(1).to(device)
    
    # data=np.concatenate((data,np.zeros((data.shape[0],3000))),axis=1)
    
    data=torch.from_numpy(np.reshape(data.astype(np.float32), (1,data.shape[0],data.shape[1]))).to(device)
 
 
    score = model(data,lens) 
    
    score=score.detach().cpu().numpy()[0,:]
    label = score>t
    
    label=label[np.array(order)]
    score=score[np.array(order)]
    
    
    current_label[label] = 1
    
    for i in range(num_classes):
        current_score[i] = np.array(score[i])
    

    return current_label, current_score




def load_12ECG_model():
    # load the model from disk 
    model_name='best_models' + os.sep  + '61_1e-05_train_0.9286569_valid_0.8222659.pkl'
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    loaded_model = torch.load(model_name,map_location=device)
    
    loaded_model=loaded_model.eval().to(device)


    return loaded_model