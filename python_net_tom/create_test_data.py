import os
import numpy as np
import json
from shutil import copyfile

test_data_folder='../../test_data'

try:
    os.mkdir(test_data_folder)
except:
    pass 


PATHS = {"labels": "../Partitioning/data/partition/",
         "data": "../../Training_WFDB/",
         }




def get_partition_data(file_name, file_path):
    with open(os.path.join(file_path, file_name)) as json_data:
        return json.load(json_data)
    
    
partition = get_partition_data("partition_82.json", PATHS["labels"])


list_of_ids=partition["validation"]

path=PATHS["data"]

for k,file_name in enumerate(list_of_ids):
    file_name=path + os.sep + file_name
    
    file_name_data=file_name+'.mat'
    file_name_lbl=file_name+'.hea'
    
    
    file_name_data_save=file_name_data.replace(path,test_data_folder)
    file_name_lbl_save=file_name_lbl.replace(path,test_data_folder)
    
    
    
    copyfile(file_name_data,file_name_data_save)

    copyfile(file_name_lbl,file_name_lbl_save)
    
    
    


