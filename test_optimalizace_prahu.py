import glob
import numpy as np
from utils.compute_challenge_metric_custom import compute_challenge_metric_custom
from utils.datareader import DataReader
from utils.utils import snomed2hot
from config import Config
from bayes_opt import BayesianOptimization


input_directory= '../data'


file_list = glob.glob(input_directory + r"\**\*.hea", recursive=True)
num_files = len(file_list)



dataReader=DataReader()
lbls_all=[]
res_all=[]


for file_num,filename in enumerate(file_list[::10]):

    print(file_num)
    
    sampling_frequency, resolution, age, sex, snomed_codes, labels=dataReader.read_header(filename,Config.SNOMED_TABLE)

    classes=list(Config.HASH_TABLE[0].keys())
    labels=snomed2hot(snomed_codes,classes)[:,0]
    
    lbls_all.append(labels)
    
    
    
    res_all.append(np.random.rand(labels.shape[0]))
    
    
    
res_all,lbls_all=np.array(res_all),np.array(lbls_all)   
    

challenge_metric_05=compute_challenge_metric_custom(res_all>0.5,lbls_all)

print(challenge_metric_05)



def aply_ts(res_all,ts):
    res_binar=np.zeros(res_all.shape,dtype=np.bool)
    for class_num,t in enumerate(ts.values()):
        res_binar[:,class_num]=res_all[:,class_num]>t
        
    return res_binar
    

def evaluate_ts(normalize=False,**ts):
    
    res_binar=aply_ts(res_all,ts)
        
    challenge_metric=compute_challenge_metric_custom(res_binar,lbls_all,normalize=normalize)
    
    return challenge_metric






func = evaluate_ts  

param_names=['t' + str(k) for k in range(lbls_all.shape[1])]
bounds_lw=0*np.ones(lbls_all.shape[1])
bounds_up=1*np.ones(lbls_all.shape[1])


pbounds=dict(zip(param_names, zip(bounds_lw,bounds_up)))

optimizer = BayesianOptimization(f=func,pbounds=pbounds,random_state=1)  
  
optimizer.maximize(init_points=250,n_iter=50)


ts=optimizer.max['params']


res_binar=aply_ts(res_all,ts)
        
challenge_metric=compute_challenge_metric_custom(res_binar,lbls_all,normalize=True)

print(challenge_metric)


