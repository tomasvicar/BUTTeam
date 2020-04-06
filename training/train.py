import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim
from dataset_class import Dataset
from torch.utils import data

import net
from train_fcns import get_best_ts,Log,get_lr,wce,beta_loss
from config import Config





def train(names_train,names_valid,model_num):
    

    device = torch.device("cuda:0")


    lbl_counts=np.load(Config.info_save_dir+os.sep+'lbl_counts.npy')
    num_of_sigs=np.load(Config.info_save_dir+os.sep+'num_of_sigs.npy')
    w_positive=num_of_sigs/lbl_counts
    w_negative=num_of_sigs/(num_of_sigs-lbl_counts)
    w_positive_tensor=torch.from_numpy(w_positive.astype(np.float32)).to(device)
    w_negative_tensor=torch.from_numpy(w_negative.astype(np.float32)).to(device)
    
    
    training_generator = Dataset(names_train, Config.DATA_PATH,'train')
    training_generator = data.DataLoader(training_generator,batch_size=Config.train_batch_size,num_workers= Config.train_num_workers, shuffle=True,drop_last=True,collate_fn=Dataset.collate_fn)

    validation_generator = Dataset(names_valid, Config.DATA_PATH,'valid')
    validation_generator = data.DataLoader(validation_generator,batch_size=Config.valid_batch_size, num_workers=Config.valid_num_workers, shuffle=False,drop_last=False,collate_fn=Dataset.collate_fn)


    if Config.pretrained==None:
        model = net.Net_addition_grow(levels=Config.levels,lvl1_size=4,input_size=12,output_size=9,convs_in_layer=3,init_conv=4,filter_size=13)
    else:
        model=torch.load(Config.pretrained) 
    model=model.to(device)


    optimizer = optim.Adam(model.parameters(),lr =Config. init_lr ,betas= (0.9, 0.999),eps=1e-8,weight_decay=1e-8)
    scheduler=optim.lr_scheduler.StepLR(optimizer, Config.step_size, gamma=Config.gamma, last_epoch=-1)
    
    if Config.loss_fcn=='wce':
        loss_fcn=wce
    elif Config.loss_fcn=='beta_loss':
        loss_fcn=beta_loss
    
    log=Log()
    
    for epoch in range(Config.max_epochs):
        
        model.train()
        for pad_seqs,lens,lbls in training_generator:
            
            pad_seqs,lens,lbls = pad_seqs.to(device),lens.to(device),lbls.to(device)

            res=model(pad_seqs,lens)
            
            loss=loss_fcn(res,lbls,w_positive_tensor,w_negative_tensor)

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            log.save_tmp_log(lbls,res,loss)
      
        
        log.save_log_data_and_clear_tmp('train')  
            
            


        
        model.eval() 
        for pad_seqs,lens,lbls in validation_generator:

            pad_seqs,lens,lbls = pad_seqs.to(device),lens.to(device),lbls.to(device)

            res=model(pad_seqs,lens)
            
            loss=loss_fcn(res,lbls,w_positive_tensor,w_negative_tensor)

            
            log.save_tmp_log(lbls,res,loss)
            
            
        log.save_log_data_and_clear_tmp('valid')     
        
            
        model.set_t(log.t)   
                
        lr=get_lr(optimizer)
        
        info= str(model_num) + '___' +str(epoch) + '_' + str(lr) + '_train_'  + str(log.trainig_beta_log[-1]) + '_valid_' + str(log.valid_beta_log[-1]) 
        print(info)
        
        model_name=Config.model_save_dir+ os.sep + Config.model_note + info  + '.pkl'
        log.save_log_model_name(model_name)
        model.save_log(log)
        torch.save(model,model_name)
            
        if Config.ploting:
            model.plot_training()
        
        scheduler.step()
        

    return log
