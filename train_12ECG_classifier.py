
import os
import glob
import numpy as np
import scipy.io as io
import dataset
import loader
import glob
import numpy as np
from config import Config
from log import Log

def train_12ECG_classifier(input_directory, output_directory):
    

    
   
    file_list = glob.glob(input_directory + r"\**\*.mat", recursive=True)
    num_files = len(file_list)
    # Train-Test split
    np.random.seed(666)
    split_ratio_ind = int(np.floor(Config.SPLIT_RATIO[0] / (Config.SPLIT_RATIO[0] + Config.SPLIT_RATIO[1]) * num_files))
    permuted_idx = np.random.permutation(num_files)
    train_ind = permuted_idx[:split_ratio_ind]
    valid_ind = permuted_idx[split_ratio_ind:]
    partition = {"train": [file_list[file_idx] for file_idx in train_ind],
        "valid": [file_list[file_idx] for file_idx in valid_ind]}
    
    
    # Train dataset generator
    training_set = dataset.Dataset( partition["train"],transform=Config.TRANSFORM_DATA_TRAIN,encode=Config.TRANSFORM_LBL)
    training_generator = data.DataLoader(training_set,batch_size=Config.BATCH_TRAIN,num_workers=Config.TRAIN_NUM_WORKERS,shuffle=True,drop_last=True,collate_fn=Dataset.collate_fn )
    
    
    validation_set = dataset.Dataset(partition["valid"],transform=Config.TRANSFORM_DATA_VALID,encode=Config.TRANSFORM_LBL)
    validation_generator = data.DataLoader(validation_set,batch_size=Config.BATCH_VALID,num_workers=num_workers=Config.VALID_NUM_WORKERS,shuffle=True,drop_last=True,collate_fn=Dataset.collate_fn )
    
    
    
    
    
    
    
    
    
    
    
    
    
    





