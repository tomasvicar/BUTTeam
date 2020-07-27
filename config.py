from utils.utils import load_weights
from utils.losses import wce,challange_metric_loss
import torch
from utils.datareader import DataReader
from utils import transforms



class Config:
    
    DATA_DIR='../data'
    
    MODEL_NOTE='test0'
    
    MODEL_SAVE_DIR='../models_tmp'
    
    SPLIT_RATIO=[9,1]

    TRAIN_NUM_WORKERS=8
    VALID_NUM_WORKERS=8
    
    BATCH_TRAIN=128
    BATCH_VALID=BATCH_TRAIN ## sould be same
    
    MAX_EPOCH=71
    STEP_SIZE=33
    GAMMA=0.1
    INIT_LR=0.01
    
    DEVICE=torch.device("cuda:0")
    
    LOSS_FCN=wce
    # LOSS_FCN=challange_metric_loss
    
    LEVELS=6
    LVL1_SIZE=4
    INPUT_SIZE=12
    OUTPUT_SIZE=24
    CONVS_IN_LAYERS=3
    INIT_CONV=4
    FILTER_SIZE=3
    
    T_OPTIMIZE_INIT=270
    T_OPTIMIZER_GP=30
    

    HASH_TABLE=DataReader.get_label_maps(path="tables/")
    SNOMED_TABLE = DataReader.read_table(path="tables/")
    SNOMED_24_ORDERD_LIST=list(HASH_TABLE[0].keys())
    
    
    output_sampling=125
    
    TRANSFORM_DATA_TRAIN=transforms.Compose([
        transforms.Resample(output_sampling=output_sampling, gain=1),
        transforms.BaseLineFilter(window_size=1000),
        transforms.ZScore(mean=0,std=1000),
        transforms.RandomShift(p=0.8),
        transforms.RandomAmplifier(p=0.8,max_multiplier=0.2),
        transforms.RandomStretch(p=0.8, max_stretch=0.2),
        ])
    
    TRANSFORM_DATA_VALID=transforms.Compose([
        transforms.Resample(output_sampling=output_sampling, gain=1),
        transforms.BaseLineFilter(window_size=1000),
        transforms.ZScore(mean=0,std=1000),
        ])
    
    TRANSFORM_LBL=transforms.SnomedToOneHot()
    
    
    
    loaded_weigths=load_weights('weights.csv',SNOMED_24_ORDERD_LIST)


