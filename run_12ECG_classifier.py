import numpy as np
import torch
import os
from utils.datareader import DataReader
from config import Config
from utils.optimize_ts import aply_ts

def run_12ECG_classifier(data,header_data,model):
    
    snomed_table = DataReader.read_table(path="tables/")
    header = DataReader.read_header(header_data, snomed_table,from_file=False)
    sampling_frequency, resolution, age, sex, snomed_codes, labels = header
    
    transform=Config.TRANSFORM_DATA_VALID
    if transform:
        data0 = transform(data, input_sampling=sampling_frequency)

    lens = data.shape[1]
    
    
    lens_all=model.lens
    batch=Config.BATCH_VALID
    
    lens_sample=np.random.choice(lens_all, batch, replace=False)
    max_len=np.max(lens_sample) 
    data_new=np.zeros((data0.shape[0],max(max_len,data0.shape[1])))
    data_new[:,:data0.shape[1]]=data0
    data_np=data_new.copy()
    
    lens=torch.from_numpy(np.array(lens).astype(np.float32)).view(1)
    data=torch.from_numpy(np.reshape(data_np.astype(np.float32), (1,data_np.shape[0],data_np.shape[1])))
    
    cuda_check = next(model.parameters()).is_cuda
    if cuda_check:
        cuda_device = next(model.parameters()).get_device()
        device = torch.device('cuda:' + str(cuda_device) )
        lens=lens.to(device)
        data=data.to(device)
        
        
    score = model(data,lens)
    score=score.detach().cpu().numpy()
    
    label = aply_ts(score,model.get_ts())
    
    score=score[0,:]
    label=label[0,:].astype(int)
    classes=list(Config.HASH_TABLE[0].keys())
    
    
    return label, score , classes


def load_12ECG_model(input_directory):
    
    f_out='model.pt'
    filename = os.path.join(input_directory,f_out)
    
    # device = torch.device("cuda:"+str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0")
    
    
    loaded_model = torch.load(filename,map_location=device)
    
    loaded_model=loaded_model.eval().to(device)
    
    
    return loaded_model