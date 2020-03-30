import os
import json
import torch
from dataset_class import Dataset
import model_classes
from torch.utils import data

PATHS = {"labels": "../data/partition/",
         "data": "../data/cinc2020/",
         }


def get_partition_data(file_name, file_path):
    with open(os.path.join(file_path, file_name)) as json_data:
        return json.load(json_data)


def train():

    # CUDA for PyTorch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    # Parameters
    params = {"batch_size": 64,
              "shuffle": True,
              "num_workers": 6}
    max_epochs = 100

    # Datasets
    partition = get_partition_data("partition_64.json", PATHS["labels"])
    labels = get_partition_data("labels.json", PATHS["labels"])

    # Generators
    training_set = Dataset(partition["train"], labels, PATHS["data"])
    training_generator = data.DataLoader(training_set, **params)

    validation_set = Dataset(partition["validation"], labels, PATHS["data"])
    validation_generator = data.DataLoader(validation_set, **params)

    # Model import
    model = model_classes.AlexNet()

    # Loop over epochs
    for epoch in range(max_epochs):
        # Training
        for local_batch, local_labels in training_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            [...]

        # Validation
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in validation_generator:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                # Model computations
                [...]


def main():
    train()


if __name__ == "__main__":
    main()