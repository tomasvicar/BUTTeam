from config import Config
import numpy as np
import os
from get_splits_means_stds_lens_counts import get_splits_means_stds_lens_counts
from train import train
from shutil import copyfile

def main():
    
    ## prepare data statistics for normalization and create random splits for crossvalidation
    # get_splits_means_stds_lens_counts()
    
    ## load crossvalidation train/valid data splits
    splits=np.load(Config.info_save_dir + os.sep +'splits.npy',allow_pickle=True)
    
    try:
        os.mkdir(Config.best_models_dir)
    except:
        pass
    
    try:
        os.mkdir(Config.model_save_dir)
    except:
        pass
    
    
    ## train each model (each crossvalidation fold), for each model select best version based on validation performance form log
    ## copy this model to best_models-folder and save its name to best_models - list
    best_models=[]
    models_performance=[]
    for model_num,split in enumerate(splits):
        print('start_training_' + str(model_num))
        
        ## train model with selected data split
        log=train(split['train'],split['valid'],model_num)
        
        ## find best model in training history based on validation performance
        best_model_ind=np.argmax(log.valid_beta_log)
        best_model_name= log.model_names[best_model_ind]   
        best_model_name_new=best_model_name.replace(Config.model_save_dir,Config.best_models_dir)   
             
        ## copy best model to special folder         
        copyfile(best_model_name, best_model_name_new)
    
        best_models.append(best_model_name_new)
        models_performance.append(np.max(log.valid_beta_log))
    
    
    
    
    ## save list with names of models - this will be used to load them
    np.save(Config.best_models_dir  + os.sep +Config.model_note + '__' + str(np.mean(models_performance)) + '.npy',best_models)
    
    
    print('final___' + str(np.mean(models_performance)))
    
    

if __name__ == "__main__":
    main()





