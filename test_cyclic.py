
import os
import glob
import numpy as np
import glob
import numpy as np
from torch import optim
from torch.utils import data as dataa
import torch
from shutil import copyfile,rmtree
from datetime import datetime
import logging
import sys


from utils.utils import get_lr
from utils.collate import PaddedCollate
from config import Config
from utils.log import Log
from utils.compute_challenge_metric_custom import compute_challenge_metric_custom
from utils.optimize_ts import optimize_ts,aply_ts
from dataset import Dataset
import net
from utils.utils import AdjustLearningRateAndLossCyclyc
from utils.get_stats import get_stats

from run_12ECG_classifier import run_12ECG_classifier,load_12ECG_model
from driver import load_challenge_data,save_challenge_predictions


# from evaluate_12ECG_score import evaluate_12ECG_score
# from evaluate_12ECG_score_fixed import evaluate_12ECG_score
from evaluate_12ECG_score_fixed_nan import evaluate_12ECG_score
import matplotlib.pyplot as plt


model = net.Net_addition_grow(levels=Config.LEVELS,
                                  lvl1_size=Config.LVL1_SIZE,
                                  input_size=Config.INPUT_SIZE,
                                  output_size=Config.OUTPUT_SIZE,
                                  convs_in_layer=Config.CONVS_IN_LAYERS,
                                  init_conv=Config.INIT_CONV,
                                  filter_size=Config.FILTER_SIZE)

optimizer = optim.Adam(model.parameters(),lr =Config.LR_LIST[0] ,betas= (0.9, 0.999),eps=1e-8,weight_decay=1e-8)
N=600
scheduler=AdjustLearningRateAndLossCyclyc(optimizer,Config.LR_LIST,Config.LR_CHANGES_LIST,Config.LOSS_FUNTIONS,N,Config.max_multiplier,Config.step_size)
    

lrs=[]
its=[]
i=0
for epoch in range(Config.MAX_EPOCH):
    for it in range(N):
        i=i+1
        its.append(i)
        
        lrs.append(get_lr(optimizer))
        
        scheduler.step()
    
    
plt.plot(np.array(its)/N,np.array(lrs))