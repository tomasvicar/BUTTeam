from utils.datareader import DataReader
import glob
from config import Config
import wfdb
import numpy as np

file_list = glob.glob(r'D:\vicar\cinc_tmp2\data_nofold_resfix/*.mat', recursive=True)

permuted_idx = np.random.permutation(len(file_list))
file_list = [file_list[file_idx] for file_idx in permuted_idx]

snomed_table = DataReader.read_table(path="tables/")
idx_mapping, label_mapping = DataReader.get_label_maps(path="tables/")
transform = Config.TRANSFORM_DATA_VALID
encode = Config.TRANSFORM_LBL

ages=[]
sexes=[]
for file_ind,file in enumerate(file_list):
    if file_ind%100==0:
        print(file_ind)
    

    signals, fields = wfdb.rdsamp(file[:-4])

    
    

    sample_file_name = file
    header_file_name = file[:-3] + "hea"

    # Read data
    # sample = DataReader.read_sample(sample_file_name).T
    header = DataReader.read_header(header_file_name, snomed_table)
    sampling_frequency, resolution, age, sex, snomed_codes, labels = header
    
    print(age,sex)
    
    ages.append(age)
    sexes.append(sex)
    
    # sample=sample.astype(np.float32)
    
    
    
    # for k in range(sample.shape[1]):
    #     sample[:,k]=sample[:,k]/resolution[k]


    # if any(np.abs(np.nanmean(sample/signals,axis=0)-1)>0.1):
    #     print(file)


    # sample = transform(sample, input_sampling=sampling_frequency)
    
    # print(resolution)
    # print(np.nanmean(sample/signals,axis=0))
    