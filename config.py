from utils.utils import load_weights
from utils.losses import wce,challange_metric_loss,FocalLoss
import torch
from utils.datareader import DataReader
from utils import transforms
import numpy as np


class Config:
    
    MODEL_NOTE='test0'
    
    
    SPLIT_RATIO=[9,1]

    TRAIN_NUM_WORKERS=6
    VALID_NUM_WORKERS=6
    
    BATCH_TRAIN=32
    BATCH_VALID=BATCH_TRAIN ## sould be same
    
    
    pretrain_whole=True
    
    
    # MODELS_SEEDS=[42+5455115,666+848489,69+448414,13+4848494,142857+849484]
    
    MODELS_SEEDS=[42+5455115,666+848489,69+448414]
    
    # MODELS_SEEDS=[42]



    LR_LIST_INIT=np.array([0.01,0.001,0.0001])/10
    LR_CHANGES_LIST_INIT=[30,20,10]
    LOSS_FUNTIONS_INIT=[wce,wce,wce]
    
    MAX_EPOCH_INIT=np.sum(LR_CHANGES_LIST_INIT)
    
    
    
    LR_LIST=np.array([0.01,0.001,0.0001])/10
    LR_CHANGES_LIST=[15,10,10]
    LOSS_FUNTIONS=[challange_metric_loss,challange_metric_loss,challange_metric_loss]
    
    MAX_EPOCH=np.sum(LR_CHANGES_LIST)
    

    
    
    DEVICE=torch.device("cuda:"+str(torch.cuda.current_device()))
    

    
    REMAP=False
    
    weight_decay=1e-5
    
    SWA=False
    SWA_NUM_EPOCHS=5
    SWA_IT_FREQ=200
    

    SEX_AND_AGE=False
    
    if SEX_AND_AGE:
        INPUT_SIZE=14
    else:
        INPUT_SIZE=12

    LEVELS=6
    LVL1_SIZE=6*5
    OUTPUT_SIZE=24
    CONVS_IN_LAYERS=3
    INIT_CONV=LVL1_SIZE
    FILTER_SIZE=7
    
    T_OPTIMIZE_INIT=250
    T_OPTIMIZER_GP=50
    

    HASH_TABLE=DataReader.get_label_maps(path="tables/")
    SNOMED_TABLE = DataReader.read_table(path="tables/")
    SNOMED_24_ORDERD_LIST=list(HASH_TABLE[0].keys())
    
    
    output_sampling=125
    std=0.2
    
    TRANSFORM_DATA_TRAIN=transforms.Compose([
        transforms.Resample(output_sampling=output_sampling),
        transforms.ZScore(mean=0,std=std),
        transforms.RandomAmplifier(p=0.8,max_multiplier=0.2),
        transforms.RandomStretch(p=0.8, max_stretch=0.1),
        ])
    
    TRANSFORM_DATA_VALID=transforms.Compose([
        transforms.Resample(output_sampling=output_sampling),
        transforms.ZScore(mean=0,std=std),
        ])
    
    TRANSFORM_LBL=transforms.SnomedToOneHot()
    
    
    
    loaded_weigths=load_weights('weights.csv',SNOMED_24_ORDERD_LIST)


