from config import Config
import numpy as np
import os
from get_splits_means_stds_lens_counts import get_splits_means_stds_lens_counts
from train import train
from shutil import copyfile

def main(): 
    get_splits_means_stds_lens_counts()
    
    splits=np.load(Config.info_save_dir + os.sep +'splits.npy',allow_pickle=True)
    
    try:
        os.mkdir(Config.best_models_dir)
    except:
        pass
    
    try:
        os.mkdir(Config.model_save_dir)
    except:
        pass
    
    best_models=[]
    models_performance=[]
    for model_num,split in enumerate(splits):
        print('start_training_' + str(model_num))
        
        log=train(split['train'],split['valid'],model_num)
        
        best_model_ind=np.argmax(log.valid_beta_log)
        best_model_name= log.model_names[best_model_ind]   
        best_model_name_new=best_model_name.replace(Config.model_save_dir,Config.best_models_dir)   
                      
        copyfile(best_model_name, best_model_name_new)
    
        best_models.append(best_model_name_new)
        models_performance.append(np.max(log.valid_beta_log))
    
    
    print('final___' + str(np.mean(models_performance)))
    
    np.save(Config.best_models_dir  + os.sep +Config.model_note + '__' + str(np.mean(models_performance)) + '.npy',best_models)





                
if __name__=='__main__':
    main()







