from utils.datareader import DataReader
import glob
from config import Config
import wfdb
import numpy as np

file_list = glob.glob(r'D:\vicar\cinc_tmp2\sample_data/*.mat', recursive=True)


snomed_table = DataReader.read_table(path="tables/")
idx_mapping, label_mapping = DataReader.get_label_maps(path="tables/")
transform = Config.TRANSFORM_DATA_VALID
encode = Config.TRANSFORM_LBL


for file in file_list:
    

    signals, fields = wfdb.rdsamp(file[:-4])

    
    

    sample_file_name = file
    header_file_name = file[:-3] + "hea"

    # Read data
    sample = DataReader.read_sample(sample_file_name).T
    header = DataReader.read_header(header_file_name, snomed_table)

    sampling_frequency, resolution, age, sex, snomed_codes, labels = header


    # sample = transform(sample, input_sampling=sampling_frequency)
    
    print(resolution)
    print(np.nanmean(sample/signals,axis=0))

