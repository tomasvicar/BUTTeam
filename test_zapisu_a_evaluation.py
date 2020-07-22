from driver import save_challenge_predictions
from read_file import read_data,LabelReader
import os
from config import Config
from utils import snomed2hot
from shutil import copyfile
import numpy as np
from evaluate_12ECG_score import evaluate_12ECG_score

output_directory='../test'
label_directory='../test_gt'
try:
    os.mkdir(output_directory)
except:
    pass
try:
    os.mkdir(label_directory)
except:
    pass

labelReader=LabelReader()


for k in range(1,9):

    filename=r"C:\Users\Tom\Desktop\tmp2_cinc2020\data\Training_WFDB\A000"+str(k)+'.hea'
    
    frequency, length,resolution, age, sex,snomed,abbreviations=labelReader.read_lbl(filename[:-4])
    
    orig_filename=filename
    
    path,filename=os.path.split(filename)
    
    copyfile(orig_filename,label_directory + os.sep+ filename )
    
    labels=snomed2hot(snomed,Config.HASH_TABLE['snomeds'])
    print(labels.T)
    
    scores=labels
    
    classes=Config.HASH_TABLE['snomeds']
    
    save_challenge_predictions(output_directory,filename.replace('.hea','.mat'),scores,labels,classes)

    


auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure, challenge_metric=evaluate_12ECG_score(label_directory, output_directory)




print(challenge_metric)



