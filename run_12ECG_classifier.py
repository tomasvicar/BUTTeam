import numpy as np
import torch
import os
from utils.datareader import DataReader
from config import Config
from utils.optimize_ts import aply_ts


def run_12ECG_classifier(data,header_data,models,traning_to_nan=False,file_name=None):

    snomed_table = DataReader.read_table(path="tables/")
    header = DataReader.read_header(header_data, snomed_table,from_file=False)
    sampling_frequency, resolution, age, sex, snomed_codes, labels = header

    transform=Config.TRANSFORM_DATA_VALID
    if transform:
        data = transform(data, input_sampling=sampling_frequency,gain=1/np.array(resolution))

    


    lens_all=models[0].lens
    batch=models[0].config.BATCH_VALID

    # Get random sample length within whole data set
    lens_sample = np.random.choice(lens_all, batch, replace=False)
    random_batch_length = np.max(lens_sample)

    # Generate batch if sample is too long, batch with random zero-padding otherwise
    reshaped_data,lens = generate_batch(data, random_batch_length,models[0].config.output_sampling)
    print(reshaped_data.shape)
    data = torch.from_numpy(reshaped_data.copy())

    
    lens = torch.from_numpy(np.array(lens).astype(np.float32))

    cuda_check = next(models[0].parameters()).is_cuda
    if cuda_check:
        cuda_device = next(models[0].parameters()).get_device()
        device = torch.device('cuda:' + str(cuda_device) )
        lens=lens.to(device)
        data=data.to(device)

    all_score=[]
    all_label =[]
    for model_num,model in enumerate(models):
        score = model(data,lens)
        score=score.detach().cpu().numpy()
        label = aply_ts(score,model.get_ts()).astype(np.float32)
    
        if traning_to_nan:
            partition=model.train_names
            
            if file_name in partition['train']:
                score[:]=np.nan
                label[:]=np.nan
    

        
        label =merge_labels(label)
        score =merge_labels(score)
    
        
        all_score.append(score)
        all_label.append(label)
        
    
    score=np.nanmean(np.array(all_score),0)
    label=np.nanmean(np.array(all_label),0)
        
        
    label=label.astype(int)    
    classes=Config.SNOMED_24_ORDERD_LIST


    return label, score , classes


def load_12ECG_model(input_directory):
    

    device = torch.device("cuda:"+str(torch.cuda.current_device()))


    f_out='model0.pt'
    filename = os.path.join(input_directory,f_out)
    loaded_model = torch.load(filename,map_location=device)


    models=[]
    for model_num in range(len(loaded_model.config.MODELS_SEEDS)):
        

        f_out='model'+ str(model_num) +'.pt'
        filename = os.path.join(input_directory,f_out)
    

        # device = torch.device("cuda:0")
    
    
        loaded_model = torch.load(filename,map_location=device)
    
        loaded_model=loaded_model.eval().to(device)

        models.append(loaded_model)

    return models


def generate_batch(sample, random_batch_length,sampling_freq):

    lens=[]

    # Compute number of chunks
    if sample.shape[1]>int(sampling_freq * 105):
    
        max_chunk_length = int(sampling_freq * 90)
        overlap = int(sampling_freq * 5)
        num_of_chunks = (sample.shape[1] - overlap) // (max_chunk_length - overlap)

    
        # Generate split indices
        onsets_list = [idx * (max_chunk_length - overlap) for idx in range(num_of_chunks)]
        offsets_list = [max_chunk_length + idx * (max_chunk_length - overlap) for idx in range(num_of_chunks)]
        offsets_list[-1] = sample.shape[1]
        max_length = max(random_batch_length, offsets_list[-1] - onsets_list[-1])

        # Initialize batch
        batch = np.zeros([num_of_chunks, 12, max_length])

        # Generate batch from sample chunks
        for idx, onset, offset in zip(range(num_of_chunks), onsets_list, offsets_list):
            chunk = sample[:, onset:offset]
            batch[idx, :, :chunk.shape[1]] = chunk
            lens.append(chunk.shape[1])

    else:
        max_length = max(random_batch_length, sample.shape[1])

        # Initialize batch
        batch = np.zeros([1, 12, max_length])
        # Generate batch
        batch[0, :, :sample.shape[1]] = sample
        
        lens.append(sample.shape[1])

    return batch.astype(np.float32),lens


def merge_labels(labels):
    """
    Merges labels across single batch
    :param labels: one hot encoded labels, shape=(batch, 1, num_of_classes)
    :return: aggregated one hot encoded labels, shape=(1, 1, num_of_classes)
    """
    return np.max(labels, axis=0)
