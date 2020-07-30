from utils.datareader import DataReader
from utils import transforms
from utils import transforms
import numpy as np
from config import Config


def get_stats(file_names_list):
    
    snomed_table = DataReader.read_table(path="tables/")
    idx_mapping,label_mapping = DataReader.get_label_maps(path="tables/")
    
    transform=Config.TRANSFORM_DATA_VALID
    encode=Config.TRANSFORM_LBL
    
    
    one_hots=[]
    lens=[]
    for idx in range(len(file_names_list)):
        sample_file_name = file_names_list[idx]
        header_file_name = file_names_list[idx][:-3] + "hea"
    
        # Read data
        sample = DataReader.read_sample(sample_file_name)
        header = DataReader.read_header(header_file_name, snomed_table)
        
        sampling_frequency, resolution, age, sex, snomed_codes, labels = header
        
        
        sample = transform(sample, input_sampling=sampling_frequency)

        sample_length = sample.shape[1]

        y = encode(snomed_codes, idx_mapping)
        
        one_hots.append(y)
        lens.append(sample_length)
        
    one_hots=np.array(one_hots)
    lens=np.array(lens)
    lbl_counts=np.sum(one_hots,0)
    
    return lbl_counts,lens
        
        
        
