import transforms
import table

class Config:
    
    DATA_PATH='../data'
    
    SPLIT_RATIO=[9,1]

    VALID_NUM_WORKERS=0
    VALID_NUM_WORKERS=0
    
    BATCH_TRAIN=32
    BATCH_VALID=8
    
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