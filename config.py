from utils import load_weights
from utils import wce
import torch
from read_file import read_table_used




class Config:
    
    
    SPLIT_RATIO=[9,1]

    VALID_NUM_WORKERS=0
    VALID_NUM_WORKERS=0
    
    BATCH_TRAIN=32
    BATCH_VALID=32
    
    MAX_EPOCH=107
    STEP_SIZE=35
    GAMMA=0.1
    INIT_LR=0.01
    
    DEVICE=torch.device("cuda:0")
    
    LOSS_FCN=wce
    
    LEVELS=8
    LVL1_SIZE=4
    INPUT_SIZE=12
    OUTPUT_SIZE=9
    CONVS_IN_LAYERS=2
    INIT_CONV=4
    FILTER_SIZE=13
    

    HASH_TABLE=read_table_used()
    
    
    # TRANSFORM_DATA_TRAIN=transforms.Compose([
    #     transforms.ZScore(mean=0, std=1),
    #     transforms.HardClip(threshold=2500),
    #     transforms.RandomVerticalFlip(p=0.2),
    #     ])
    
    # TRANSFORM_DATA_VALID=transforms.Compose([
    #     transforms.ZScore(mean=0, std=1),
    #     transforms.HardClip(threshold=2500),
    #     ])
    
    # TRANSFORM_LBL=transforms.Compose([
    #     transforms.OneHot(table),
    #     ])
    
    
    
    loaded_weigths=load_weights('weights.csv',HASH_TABLE['snomeds'])


