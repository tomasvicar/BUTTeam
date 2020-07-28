
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


