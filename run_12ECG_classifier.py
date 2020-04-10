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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        
        
        lens_sample=np.random.choice(lens_all, batch, replace=False)
        max_len=np.max(lens_sample)
        
        data_new=np.zeros((data0.shape[0],max(max_len,data0.shape[1])))
        data_new[:,:data0.shape[1]]=data0
        
        data_np=data_new.copy()
        
        
        
        data_np=data0.copy()
        
        
        lens=data0.shape[1]
        lens=torch.from_numpy(np.array(lens).astype(np.float32)).view(1).to(device)
        
        data=torch.from_numpy(np.reshape(data_np.astype(np.float32), (1,data_np.shape[0],data_np.shape[1]))).to(device)
        
        
        score = model(data,lens) 
        
        score=score.detach().cpu().numpy()[0,:]
        
        label = (score>model.get_t()[0,:])
        # label = (score>0.5)
        
        score=score[np.array(order)]
        label=label[np.array(order)]
        
        
        for i in range(num_classes):
            current_score[model_num,i] = np.array(score[i])
            current_label[model_num,i] = np.array(label[i])
            
            
            
            
    splits=np.load('training/data_split/splits.npy',allow_pickle=True)
    curent_name=header_data[0].split(' ')[0]
    for k in range(len(models)):
        all_names=splits[k]['train']
        counter=0
        for name in all_names:
            if curent_name==name:
                counter=counter+1
                current_score[k,:]=np.nan
                current_label[k,:]=np.nan

    current_score=np.nanmean(current_score,axis=0)
    current_label=np.round(np.nanmean(current_label,axis=0)).astype(np.int)
    
    
    
    
    
    
    
    return current_label, current_score


# 0.8512470205326718
# 0.8294387709634904



def load_12ECG_model():
    # load the model from disk 
    # models_names_name='training/best_models/z6na12conv__0.75622284.npy' ##0.7132331608211755
    # models_names_name='training/best_models/conv12_8lvlu__0.75800914.npy' ####0.7224464697031729 
    models_names_name='training/best_models/no_pretrain__0.76311535.npy' ### 0.7272554585131986   
    # models_names_name='training/best_models/no_pretrain__0.7499013.npy'  ### 0.7130929532735658
    
    models=[]
    
    models_names=np.load(models_names_name,allow_pickle=True)
    
    for model_name in models_names:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        
        loaded_model = torch.load('training/' + model_name,map_location=device)
        
        loaded_model=loaded_model.eval().to(device)
        
        models.append(loaded_model)

    return models