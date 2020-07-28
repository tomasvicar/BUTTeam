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
        data = transform(data, input_sampling=sampling_frequency)

    


    lens_all=model.lens
    batch=model.config.BATCH_VALID

    # Get random sample length within whole data set
    lens_sample = np.random.choice(lens_all, batch, replace=False)
    random_batch_length = np.max(lens_sample)

    # Generate batch if sample is too long, batch with random zero-padding otherwise
    reshaped_data,lens = generate_batch(data, random_batch_length,model.config.output_sampling)
    print(reshaped_data.shape)
    data = torch.from_numpy(reshaped_data.copy())

    
    lens = torch.from_numpy(np.array(lens).astype(np.float32))

    cuda_check = next(model.parameters()).is_cuda
    if cuda_check:
        cuda_device = next(model.parameters()).get_device()
        device = torch.device('cuda:' + str(cuda_device) )
        lens=lens.to(device)
        data=data.to(device)


    score = model(data,lens)
    score=score.detach().cpu().numpy()

    label = aply_ts(score,model.get_ts())
    # label = score>0.5
    
    label =merge_labels(label)
    score =merge_labels(score)

    label=label.astype(int)
    classes=Config.SNOMED_24_ORDERD_LIST


    return label, score , classes


def load_12ECG_model(input_directory):

    f_out='model.pt'
    filename = os.path.join(input_directory,f_out)

    device = torch.device("cuda:"+str(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda:0")


    loaded_model = torch.load(filename,map_location=device)

    loaded_model=loaded_model.eval().to(device)


    return loaded_model


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
