import transforms
import table
from utils import wce

class Config:
    
    DATA_PATH='../data'
    
    SPLIT_RATIO=[9,1]

    VALID_NUM_WORKERS=0
    VALID_NUM_WORKERS=0
    
    BATCH_TRAIN=32
    BATCH_VALID=32
    
    MAX_EPOCH=107
    STEP_SIZE=35
    GAMMA=0.1
    INIT_LR=0.01
    
    LOSS_FCN=wce
    
    TRANSFORM_DATA_TRAIN=transforms.Compose([
        transforms.ZScore(mean=0, std=1),
        transforms.HardClip(threshold=2500),
        transforms.RandomVerticalFlip(p=0.2),
        ]),
    
    TRANSFORM_DATA_VALID=transforms.Compose([
        transforms.ZScore(mean=0, std=1),
        transforms.HardClip(threshold=2500),
        ])
    
    TRANSFORM_LBL=transforms.Compose([
        transforms.OneHot(table),
        ]),