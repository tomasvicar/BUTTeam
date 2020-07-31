from torch.utils import data as dataa
from utils.collate import PaddedCollate
import numpy as np
from config import Config
from dataset import Dataset
from copy import deepcopy

def get_stats(file_names_list):

    validation_set = Dataset(file_names_list,transform=Config.TRANSFORM_DATA_VALID,encode=Config.TRANSFORM_LBL)
    validation_generator = dataa.DataLoader(validation_set,batch_size=Config.BATCH_VALID,num_workers=Config.VALID_NUM_WORKERS,
                                           shuffle=False,drop_last=False,collate_fn=PaddedCollate() )
    
    
    one_hots=[]
    lenss=[]
    for it,(pad_seqs,lbls,lens) in enumerate(validation_generator):
        print(it)
        
        one_hots.append(deepcopy(lbls.detach().cpu().numpy()))
        lenss.append(deepcopy(lens.detach().cpu().numpy()))
        del lbls
        del lens
        del pad_seqs
    
        
    one_hots=np.concatenate(one_hots,axis=0)
    lenss=np.concatenate(lenss,axis=0)
    lbl_counts=np.sum(one_hots,0)
    
    return lbl_counts,lenss