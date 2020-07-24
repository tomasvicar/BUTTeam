from driver import save_challenge_predictions
from datareader import DataReader
import os
from config import Config
from utils import snomed2hot
from shutil import copyfile
import numpy as np
from evaluate_12ECG_score_fixed import evaluate_12ECG_score
from compute_challenge_metric_custom import compute_challenge_metric_custom

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

dataReader=DataReader()

lbls_all=[]
res_all=[]
np.random.seed(42)
for k in range(1,10):

    filename=r"C:\Users\Tom\Desktop\tmp2_cinc2020\data\Training_WFDB\A000"+str(k)+'.hea'
    
    frequency, length,resolution, age, sex,snomed,abbreviations=dataReader.read_header(filename[:-4])
    
    orig_filename=filename
    
    path,filename=os.path.split(filename)
    
    copyfile(orig_filename,label_directory + os.sep+ filename )
    
    classes=list(Config.HASH_TABLE[0].keys())
    labels=snomed2hot(snomed,classes)[:,0]
    
    
    
    # labels=np.concatenate((labels,labels[4:5],labels[12:13],labels[13:14]))
    
    # classes = classes + ['59118001','63593006', '17338001']

    lbls_all.append(labels)
    
    # labels[0]=1
    labels=(np.random.rand(labels.shape[0])>0.5).astype(np.float)
    
    
    print(labels)
    
    scores=labels
    
    labels=labels.astype(int)
    
    res_all.append(labels)
    

    
    save_challenge_predictions(output_directory,filename.replace('.hea','.mat'),scores,labels,classes)

    


auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure, challenge_metric=evaluate_12ECG_score(label_directory, output_directory)

print(challenge_metric)

challenge_metric2=compute_challenge_metric_custom(np.array(res_all),np.array(lbls_all))

print(challenge_metric2)