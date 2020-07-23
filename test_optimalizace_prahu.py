import glob
import numpy as np
from compute_challenge_metric_custom import compute_challenge_metric_custom
from read_file import read_data,LabelReader
from utils import snomed2hot
from config import Config

input_directory= '../data'


file_list = glob.glob(input_directory + r"\**\*.mat", recursive=True)
num_files = len(file_list)



labelReader=LabelReader()

lbls_all=[]
res_all=[]

for file_num,filename in enumerate(file_list):

    
    frequency, length,resolution, age, sex,snomed,abbreviations=labelReader.read_lbl(filename[:-4])
    labels=snomed2hot(snomed,Config.HASH_TABLE['snomeds'])[:,0]
    
    lbls_all.append(labels)
    
    
    
    res_all.append(np.random.rand(labels.shape[0]))
    
    
    
res_all,lbls_all=np.array(res_all),np.array(lbls_all)   
    

challenge_metric_05=compute_challenge_metric_custom(res_all>0.5,lbls_all)








    
    
    