#!/usr/bin/env python

import torch
import numpy as np
import sys
sys.path.append('training')

def run_12ECG_classifier(data,header_data,classes,model):
    
    
    models=model
    
    ## create empty array for saving of resutls of all mdels 
    num_classes = len(classes)
    current_label = np.zeros((len(models),num_classes))
    current_score = np.zeros((len(models),num_classes))
    

    
    ## load data statisctics for normalization
    MEANS=np.load('training/data_split/MEANS.npy')
    STDS=np.load('training/data_split/STDS.npy')
    pato_names=np.load('training/data_split/pato_names.npy')
    lens_all=np.load('training/data_split/lens.npy')
    batch=models[0].config.train_batch_size
    
    
    ## how to reorder pathologies from our to their ordering
    order=[];
    for cl in classes:
        for k,pato in enumerate(pato_names):
            if pato==cl:
                order.append(k)
    
    
    ## normalize data  - same means and stds as is used in training dataloder
    data0=(data-MEANS.reshape(-1,1))/STDS.reshape(-1,1).copy()
    
    
 
    ## iterate over crossvalidated models
    for model_num,model in enumerate(models):
        
        
        ## simulate a batch - radnomly sample training signal lens and pad with zeros to largest len
        lens_sample=np.random.choice(lens_all, batch, replace=False)
        max_len=np.max(lens_sample) 
        data_new=np.zeros((data0.shape[0],max(max_len,data0.shape[1])))
        data_new[:,:data0.shape[1]]=data0
        data_np=data_new.copy()
        
        
        # data_np=data0.copy()
        
        
        ## model require signal len for removal of padded part before max pooling
        lens=data0.shape[1]
        
        ## numpy array => tensor
        lens=torch.from_numpy(np.array(lens).astype(np.float32)).view(1)
        data=torch.from_numpy(np.reshape(data_np.astype(np.float32), (1,data_np.shape[0],data_np.shape[1])))
        

        ## if model is using cuda, then send data to cuda
        cuda_check = next(model.parameters()).is_cuda
        if cuda_check:
            cuda_device = next(model.parameters()).get_device()
            device = torch.device('cuda:' + str(cuda_device) )
            lens=lens.to(device)
            data=data.to(device)
        
        
        
        ## predict with model
        score = model(data,lens) 
        
        ## tensor => numpy array
        score=score.detach().cpu().numpy()[0,:]
        
        ## treshold to get labels - model.get_t() gives you optimal threshold
        label = (score>model.get_t()[0,:])
        # label = (score>0.5)
        
        ## change pathology order from our to their order
        score=score[np.array(order)]
        label=label[np.array(order)]
        
        
        ## save results 
        for i in range(num_classes):
            current_score[model_num,i] = np.array(score[i])
            current_label[model_num,i] = np.array(label[i])
            
            
    ## check if this sample was used for training of each model , and not to use its prediction if it was used
    
    # splits=np.load('training/data_split/splits.npy',allow_pickle=True)
    # curent_name=header_data[0].split(' ')[0]
    # for k in range(len(models)):
    #     all_names=splits[k]['train']
    #     counter=0
    #     for name in all_names:
    #         if curent_name==name:
    #             counter=counter+1
    #             current_score[k,:]=np.nan
    #             current_label[k,:]=np.nan
    


    ## use mean of all models as predictions
    current_score=np.nanmean(current_score,axis=0)
    current_label=np.round(np.nanmean(current_label,axis=0)).astype(np.int)
    
    
    
    
    
    
    
    return current_label, current_score





def load_12ECG_model():
    
 
    models_names_name='training/best_models/small_model__0.7635788.npy'
    
    models=[]
    
    # load list with model names
    models_names=np.load(models_names_name,allow_pickle=True)
    
    ## load each model and save to the list
    for model_name in models_names:
        
        # device = torch.device("cuda:"+str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        
        
        loaded_model = torch.load('training/' + model_name,map_location=device)
        
        ### set model to eval mode and send it to graphic card
        loaded_model=loaded_model.eval().to(device)
        
        models.append(loaded_model)

    return models