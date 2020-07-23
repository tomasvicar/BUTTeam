import glob
import numpy as np
from compute_challenge_metric_custom import compute_challenge_metric_custom
from read_file import read_data,LabelReader
from utils import snomed2hot
from config import Config
from bayes_opt import BayesianOptimization


input_directory= '../data'


file_list = glob.glob(input_directory + r"\**\*.mat", recursive=True)
num_files = len(file_list)



labelReader=LabelReader()

lbls_all=[]
res_all=[]

for file_num,filename in enumerate(file_list[0:20]):

    print(file_num)
    
    frequency, length,resolution, age, sex,snomed,abbreviations=labelReader.read_lbl(filename[:-4])
    labels=snomed2hot(snomed,Config.HASH_TABLE['snomeds'])[:,0]
    
    lbls_all.append(labels)
    
    
    
    res_all.append(np.random.rand(labels.shape[0]))
    
    
    
res_all,lbls_all=np.array(res_all),np.array(lbls_all)   
    

challenge_metric_05=compute_challenge_metric_custom(res_all>0.5,lbls_all)

print(challenge_metric_05)


# param_names=['min_dist','min_value','min_h','min_size']
# bounds_lw=[1,   0,      0,      20]
# bounds_up=[80,  0.9,    0.9,    400]





# pbounds=dict(zip(segmenter.param_names, zip(segmenter.bounds_lw,segmenter.bounds_up)))

# optimizer = BayesianOptimization(f=func,pbounds=pbounds,random_state=1)  
  
#   optimizer.maximize(init_points=50,n_iter=150)