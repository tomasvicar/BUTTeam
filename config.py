from utils.utils import load_weights
from utils.utils import wce
import torch
from utils.datareader import DataReader
from utils import transforms



class Config:
    
    DATA_DIR='../data'
    
    MODEL_NOTE='test0'
    
    MODEL_SAVE_DIR='../models_tmp'
    
    SPLIT_RATIO=[9,1]

    TRAIN_NUM_WORKERS=0
    VALID_NUM_WORKERS=0
    
    BATCH_TRAIN=32
    BATCH_VALID=BATCH_TRAIN ## sould be same
    
    MAX_EPOCH=2
    STEP_SIZE=10
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
    FILTER_SIZE=5
    
    T_OPTIMIZE_INIT=250
    T_OPTIMIZER_GP=1
    

    HASH_TABLE=DataReader.get_label_maps(path="tables/")
    SNOMED_TABLE = DataReader.read_table(path="tables/")
    
    
    TRANSFORM_DATA_TRAIN=transforms.Compose([
        transforms.Resample(output_sampling=500, gain=1),
        transforms.BaseLineFilter(window_size=1000),
        transforms.ZScore(mean=0,std=1000),
        transforms.RandomShift(p=0.8),
        transforms.RandomAmplifier(p=0.8,max_multiplier=0.2),
        transforms.RandomStretch(p=0.8, max_stretch=0.2),
        ])
    
    TRANSFORM_DATA_VALID=transforms.Compose([
        transforms.Resample(output_sampling=500, gain=1),
        transforms.BaseLineFilter(window_size=1000),
        transforms.ZScore(mean=0,std=1000),
        ])
    
    TRANSFORM_LBL=transforms.SnomedToOneHot()
    
    
    
    loaded_weigths=load_weights('weights.csv',list(HASH_TABLE[0].keys()))


