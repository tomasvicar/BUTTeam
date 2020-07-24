from utils.utils import load_weights
from utils.utils import wce
import torch
from utils.datareader import DataReader
from utils import transforms



class Config:
    
    MODEL_NOTE='test0'
    
    SPLIT_RATIO=[9,1]

    TRAIN_NUM_WORKERS=4
    VALID_NUM_WORKERS=2
    
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
    OUTPUT_SIZE=24
    CONVS_IN_LAYERS=2
    INIT_CONV=4
    FILTER_SIZE=13
    

    HASH_TABLE=DataReader.get_label_maps(path="tables/")
    SNOMED_TABLE = DataReader.read_table(path="tables/")
    
    
    TRANSFORM_DATA_TRAIN=transforms.Compose([
        transforms.Resample(output_sampling=500, gain=1),
        transforms.BaseLineFilter(window_size=1000),
        transforms.ZScore(mean=0,std=1000),
        ])
    
    TRANSFORM_DATA_VALID=transforms.Compose([
        transforms.Resample(output_sampling=500, gain=1),
        transforms.BaseLineFilter(window_size=1000),
        transforms.ZScore(mean=0,std=1000),
        ])
    
    TRANSFORM_LBL=transforms.SnomedToOneHot()
    
    
    
    loaded_weigths=load_weights('weights.csv',list(HASH_TABLE[0].keys()))


