import torch
import model_classes
from torch.utils import data
from torch.backends import cudnn
from buttchery import loader, dataset
import config


def train(model, partition, labels, validation=False):
    """
    Train neural net model
    """
    # CUDA setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = config.CUDA["benchmark"]

    # Train dataset generator
    training_set = dataset.Dataset(
        partition["train"],
        labels,
        config.FILES["data"],
        transform=config.TRANSFORM["data"],
        encode=config.TRANSFORM["labels"],
    )
    training_generator = data.DataLoader(
        training_set,
        **config.BATCH,
    )

    # Validation dataset generator
    if validation:
        validation_set = dataset.Dataset(
            partition["validation"],
            labels,
            config.FILES["data"],
        )
        validation_generator = data.DataLoader(
            validation_set,
            **config.BATCH,
        )

    for epoch in range(config.LEARNING["max_epochs"]):
        # Training
        for local_batch, local_labels in training_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            pass

        # Validation
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in validation_generator:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                # Model computations
                pass


def main():
    # Load partitioning files
    partition = loader.get_json(config.FILES["partition"])
    labels = loader.get_json(config.FILES["labels"])


    # Model import
    model = model_classes.AlexNet()

    # kriteriální a optimalizační funkce
    # vahy = torch.tensor([0.33, 0.34, 0.33])
    # criterion = nn.CrossEntropyLoss()
    # optim_params = {"lr": learning_rate,
    #                 "weight_decay": 0.0001,
    #                 }
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

    train(model, partition, labels)


if __name__ == "__main__":
    main()