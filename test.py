
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
# from evaluate_12ECG_score_fixed import evaluate_12ECG_score
from evaluate_12ECG_score_fixed_nan import evaluate_12ECG_score






model_input = 'model'
input_directory = '../data_nofold'
output_directory = '../results'



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

for i, f in enumerate(input_files):
    print('    {}/{}...'.format(i+1, num_files))
    tmp_input_file = os.path.join(input_directory,f)
    data,header_data = load_challenge_data(tmp_input_file)
    current_label, current_score,classes = run_12ECG_classifier(data,header_data, model,traning_to_nan=True,file_name=f)
    # Save results.
    save_challenge_predictions(output_directory,f,current_score,current_label,classes)


print('Done.')




print('evaluating')
auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure, challenge_metric=evaluate_12ECG_score(input_directory, output_directory)

print(challenge_metric)

output_file='notes/result' + datetime.now().strftime("%H_%M_%d_%m_%Y") + '.txt'
with open(output_file, 'w') as f:
    f.write(str(challenge_metric))