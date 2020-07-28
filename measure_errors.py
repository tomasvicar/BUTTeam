
import os
import glob
import numpy as np
import glob
import numpy as np
from torch import optim
from torch.utils import data as dataa
import torch
from shutil import copyfile,rmtree


from utils.utils import get_lr
from utils.collate import PaddedCollate
from config import Config
from utils.log import Log
from utils.compute_challenge_metric_custom import compute_challenge_metric_custom
from utils.optimize_ts import optimize_ts,aply_ts
from dataset import Dataset
import net
from utils.get_data_info import enumerate_labels,sub_dataset_labels_sum

from run_12ECG_classifier import run_12ECG_classifier,load_12ECG_model
from driver import load_challenge_data,save_challenge_predictions
# from evaluate_12ECG_score import evaluate_12ECG_score
from evaluate_12ECG_score_fixed import evaluate_12ECG_score

from utils.datareader import DataReader
from utils import transforms

model_input = 'model'
input_directory = '../data_tmp'
output_directory = '../results'





#####################
## modified part

try:
    rmtree(input_directory)
except:
    pass

if not os.path.isdir(input_directory):
    os.mkdir(input_directory)


file_list = glob.glob(Config.DATA_DIR + r"\**\*.mat", recursive=True)
file_list =[x for x in file_list if 'Training_StPetersburg' not in x]

num_files = len(file_list)

# Train-Test split
state=np.random.get_state()
np.random.seed(42)
split_ratio_ind = int(np.floor(Config.SPLIT_RATIO[0] / (Config.SPLIT_RATIO[0] + Config.SPLIT_RATIO[1]) * num_files))
permuted_idx = np.random.permutation(num_files)
train_ind = permuted_idx[:split_ratio_ind]
valid_ind = permuted_idx[split_ratio_ind:]
partition = {"train": [file_list[file_idx] for file_idx in train_ind],
    "valid": [file_list[file_idx] for file_idx in valid_ind]}
np.random.set_state(state)

for file_num,file in enumerate(partition['valid']):
    path,file_name=os.path.split(file)
    
    copyfile(file,input_directory + os.sep + file_name)
    copyfile(file.replace('.mat','.hea'),input_directory + os.sep + file_name.replace('.mat','.hea'))

try:
    rmtree(output_directory)
except:
    pass

##################################





# Find files.
input_files = []
for f in os.listdir(input_directory):
    if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
        input_files.append(f)


if not os.path.isdir(output_directory):
    os.mkdir(output_directory)

# Load model.
print('Loading 12ECG model...')
model = load_12ECG_model(model_input)

# Iterate over files.
print('Extracting 12ECG features...')
num_files = len(input_files)

lbls_all=[]
res_all=[]

for i, f in enumerate(input_files):
    print('    {}/{}...'.format(i+1, num_files))
    tmp_input_file = os.path.join(input_directory,f)
    data,header_data = load_challenge_data(tmp_input_file)
    
    snomed_table = DataReader.read_table(path="tables/")
    header = DataReader.read_header(header_data, snomed_table,from_file=False)
    sampling_frequency, resolution, age, sex, snomed_codes, labels = header
    
    current_label, current_score,classes = run_12ECG_classifier(data,header_data, model)
    
    idx_mapping,label_mapping = DataReader.get_label_maps(path="tables/")
    encode=transforms.SnomedToOneHot()
    y = encode(snomed_codes, idx_mapping)
    
    
    lbls_all.append(y)
    res_all.append(current_label)
    
    # Save results.
    save_challenge_predictions(output_directory,f,current_score,current_label,classes)


print('Done.')

print('evaluating')
auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure, challenge_metric=evaluate_12ECG_score(input_directory, output_directory)

print(challenge_metric)

lbls_all=np.array(lbls_all)
res_all=np.array(res_all)

from config import Config

TP=np.sum((lbls_all==1)&(res_all==1),axis=0)
FP=np.sum((lbls_all==0)&(res_all==1),axis=0)
FN=np.sum((lbls_all==1)&(res_all==0),axis=0)
TN=np.sum((lbls_all==0)&(res_all==0),axis=0)

classes=Config.SNOMED_24_ORDERD_LIST


class_string = ','.join(classes)
TP_string = ','.join(str(i) for i in TP)
FP_string = ','.join(str(i) for i in FP)
FN_string = ','.join(str(i) for i in FN)
TN_string = ','.join(str(i) for i in TN)


output_file='notes/prediction_errors.csv'
with open(output_file, 'w') as f:
    f.write(class_string +'\n' + TP_string + '\n'+'FP_string' +'\n'+FN_string +'\n'+TN_string +'\n' )
    
    
    
output_file='notes/prediction_all_order_spatne_lbls_pred.csv'
with open(output_file, 'w') as f:
    for k in range(lbls_all.shape[0]):
        
        f.write(input_files[k] +',' +','.join(str(i) for i in (lbls_all[k,:].astype(np.int)!=res_all[k,:]).astype(np.int)) + ',xxxxxx, ' + ','.join(str(i) for i in lbls_all[k,:].astype(np.int)) + ',xxxxxx,' +  ','.join(str(i) for i in res_all[k,:])  +  '\n' )
    

import matplotlib.pyplot as plt

width = 0.65
ind=np.arange(24)
p1 = plt.bar(ind, TP, width)
p2 = plt.bar(ind, FP, width,bottom=TP)
p3 = plt.bar(ind, FN, width,bottom=TP+FP)
p4 = plt.bar(ind, TN, width,bottom=TP+FP+FN)


plt.xticks(ind, classes)
plt.xticks(rotation=-45)
plt.legend((p1[0], p2[0],p3[0],p4[0]), ('TP', 'FP','FN','TN'))
plt.savefig('notes/prediction_errors_snomed.png',dpi=200)
plt.close()


width = 0.65
ind=np.arange(24)
p1 = plt.bar(ind, TP, width)
p2 = plt.bar(ind, FP, width,bottom=TP)
p3 = plt.bar(ind, FN, width,bottom=TP+FP)
p4 = plt.bar(ind, TN, width,bottom=TP+FP+FN)


plt.xticks(ind, [Config.HASH_TABLE[1][x] for x in classes])
plt.xticks(rotation=-45)
plt.legend((p1[0], p2[0],p3[0],p4[0]), ('TP', 'FP','FN','TN'))
plt.savefig('notes/prediction_errors_abb.png',dpi=200)
plt.close()



