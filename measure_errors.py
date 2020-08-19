
import os
import glob
import numpy as np
import glob
import numpy as np
from torch import optim
from torch.utils import data as dataa
import torch
from shutil import copyfile,rmtree
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
from utils.utils import AdjustLearningRateAndLoss
from utils.get_stats import get_stats

from run_12ECG_classifier import run_12ECG_classifier,load_12ECG_model
from driver import load_challenge_data,save_challenge_predictions
from utils.datareader import DataReader
from utils import transforms


# from evaluate_12ECG_score import evaluate_12ECG_score
# from evaluate_12ECG_score_fixed import evaluate_12ECG_score
from evaluate_12ECG_score_fixed_nan import evaluate_12ECG_score

# model_input = 'model'
# input_directory = '../data_tmp'
# output_directory = '../results'



# model_input = 'model'
# input_directory = '../data_nofold_resfix'
# output_directory = '../results'




# # Find files.
# input_files = []
# for f in os.listdir(input_directory):
#     if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
#         input_files.append(f)


# if not os.path.isdir(output_directory):
#     os.mkdir(output_directory)

# # Load model.
# print('Loading 12ECG model...')
# model = load_12ECG_model(model_input)

# # Iterate over files.
# print('Extracting 12ECG features...')
# num_files = len(input_files)

# lbls_all=[]
# res_all=[]

# for i, f in enumerate(input_files):
#     print('    {}/{}...'.format(i+1, num_files))
#     tmp_input_file = os.path.join(input_directory,f)
#     data,header_data = load_challenge_data(tmp_input_file)
    
#     snomed_table = DataReader.read_table(path="tables/")
#     header = DataReader.read_header_keep_snomed(header_data, snomed_table,from_file=False)
#     sampling_frequency, resolution, age, sex, snomed_codes = header
    
#     current_label, current_score,classes = run_12ECG_classifier(data,header_data, model,traning_to_nan=True,file_name=f)
    
#     idx_mapping,label_mapping = DataReader.get_label_maps(path="tables/")
#     encode=transforms.SnomedToOneHot()
#     y = encode(snomed_codes, idx_mapping)
    
    
#     lbls_all.append(y)
#     res_all.append(current_label)
    
#     # Save results.
#     save_challenge_predictions(output_directory,f,current_score,current_label,classes)


# print('Done.')

# print('evaluating')
# auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure, challenge_metric=evaluate_12ECG_score(input_directory, output_directory)

# print(challenge_metric)

# lbls_all=np.array(lbls_all)
# res_all=np.array(res_all)

# np.save('lbls_all.npy',lbls_all)
# np.save('res_all.npy',res_all)

lbls_all=np.load('lbls_all.npy')
res_all=np.load('res_all.npy')


nan_rows=np.sum(res_all==-2147483648,axis=1)==0

lbls_all=lbls_all[nan_rows,:]
res_all=res_all[nan_rows,:]


from config import Config

TP=np.sum((lbls_all==1)&(res_all==1),axis=0)
FP=np.sum((lbls_all==0)&(res_all==1),axis=0)
FN=np.sum((lbls_all==1)&(res_all==0),axis=0)
TN=np.sum((lbls_all==0)&(res_all==0),axis=0)

classes=Config.SNOMED_24_ORDERD_LIST
classes_abb=[Config.HASH_TABLE[1][x] for x in classes]

dice = (2*TP)/(2*TP + FP + FN)




class_string = ','.join(classes_abb)
TP_string = ','.join(str(i) for i in TP)
FP_string = ','.join(str(i) for i in FP)
FN_string = ','.join(str(i) for i in FN)
TN_string = ','.join(str(i) for i in TN)

dice_string = ','.join(str(i) for i in dice)


output_file='notes/prediction_errors.csv'
with open(output_file, 'w') as f:
    f.write('class,'+ class_string +'\n' + 'TP,'+ TP_string + '\n' + 'FP,' +FP_string +'\n'+ 'FN,' + FN_string +'\n'+'TN,' + TN_string +'\n' + 'dice,' +dice_string +'\n' )
    

import matplotlib.pyplot as plt



width = 0.65
ind=np.arange(24)
p1 = plt.bar(ind, TP, width)
p2 = plt.bar(ind, FP, width,bottom=TP)
p3 = plt.bar(ind, FN, width,bottom=TP+FP)
p4 = plt.bar(ind, TN, width,bottom=TP+FP+FN)


plt.xticks(ind, classes_abb)
plt.xticks(rotation=-90)
plt.legend((p1[0], p2[0],p3[0],p4[0]), ('TP', 'FP','FN','TN'))
plt.savefig('notes/prediction_errors_abb.png',dpi=200)
plt.savefig('notes/prediction_errors_abb.svg')
plt.savefig('notes/prediction_errors_abb.eps')
plt.close()

    
 


new_order=np.argsort(dice)[::-1]

TP=TP[new_order]
FP=FP[new_order]
FN=FN[new_order]
TN=TN[new_order]
dice=dice[new_order]

classes=[classes[x] for x in new_order]
classes_abb=[Config.HASH_TABLE[1][x] for x in classes]




class_string = ','.join(classes_abb)
TP_string = ','.join(str(i) for i in TP)
FP_string = ','.join(str(i) for i in FP)
FN_string = ','.join(str(i) for i in FN)
TN_string = ','.join(str(i) for i in TN)

dice_string = ','.join(str(i) for i in dice)


output_file='notes/prediction_errors_diceorder.csv'
with open(output_file, 'w') as f:
    f.write('class,'+ class_string +'\n' + 'TP,'+ TP_string + '\n' + 'FP,' +FP_string +'\n'+ 'FN,' + FN_string +'\n'+'TN,' + TN_string +'\n' + 'dice,' +dice_string +'\n' )
    

import matplotlib.pyplot as plt



width = 0.65
ind=np.arange(24)
p1 = plt.bar(ind, TP, width)
p2 = plt.bar(ind, FP, width,bottom=TP)
p3 = plt.bar(ind, FN, width,bottom=TP+FP)
p4 = plt.bar(ind, TN, width,bottom=TP+FP+FN)


plt.xticks(ind, classes_abb)
plt.xticks(rotation=-90)
plt.legend((p1[0], p2[0],p3[0],p4[0]), ('TP', 'FP','FN','TN'))
plt.savefig('notes/prediction_errors_abb_diceorder.png',dpi=200)
plt.savefig('notes/prediction_errors_abb_diceorder.svg')
plt.savefig('notes/prediction_errors_abb_diceorder.eps')
plt.close()

    
