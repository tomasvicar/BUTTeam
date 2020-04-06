#!/usr/bin/env python

import torch
import numpy as np
import os
from config import Config


def run_12ECG_classifier(data,header_data,classes,model):
    
    models=model
    
    num_classes = len(classes)
    current_label = np.zeros((len(models),num_classes))
    current_score = np.zeros((len(models),num_classes))
    

    
    
    MEANS=np.load('training/data_split/MEANS.npy')
    
    STDS=np.load('training/data_split/STDS.npy')
    
    pato_names=np.load('training/data_split/pato_names.npy')
    
    lens_all=np.load('training/data_split/lens.npy')
    
    batch=Config.train_batch_size
    
    order=[];
    for cl in classes:
        for k,pato in enumerate(pato_names):
            if pato==cl:
                order.append(k)
    
    
    
    data0=(data-MEANS.reshape(-1,1))/STDS.reshape(-1,1).copy()
    
    
 
    
    for model_num,model in enumerate(models):
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        
        lens_sample=np.random.choice(lens_all, batch, replace=False)
        max_len=np.max(lens_sample)
        
        data_new=np.zeros((data0.shape[0],max(max_len,data0.shape[1])))
        data_new[:,:data0.shape[1]]=data0
        
        
        
        data_np=data_new.copy()
        
        lens=data_np.shape[1]
        lens=torch.from_numpy(np.array(lens).astype(np.float32)).view(1).to(device)
        
        data=torch.from_numpy(np.reshape(data_np.astype(np.float32), (1,data_np.shape[0],data_np.shape[1]))).to(device)
        
        
        score = model(data,lens) 
        
        score=score.detach().cpu().numpy()[0,:]
        score=score[np.array(order)]
        
        label = (score>model.get_t()[0,:])
        
        
        for i in range(num_classes):
            current_score[model_num,i] = np.array(score[i])
            current_label[model_num,i] = np.array(label[i])

    current_score=np.mean(current_score,axis=0)
    current_label=np.round(np.mean(current_label,axis=0)).astype(np.int)
    
    return current_label, current_score




def load_12ECG_model():
    # load the model from disk 
    models_names_name='training/best_models/aug_attentintest_best_t__0.38516265.npy'
    
    models=[]
    
    models_names=np.load(models_names_name,allow_pickle=True)
    
    for model_name in models_names:
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        
        loaded_model = torch.load('training/' + model_name,map_location=device)
        
        loaded_model=loaded_model.eval().to(device)
        
        models.append(loaded_model)

    return models