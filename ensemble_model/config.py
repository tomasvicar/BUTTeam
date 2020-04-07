from buttchery import transforms, collate

# count = {'Normal': 1645, 'AF': 217, 'I-AVB': 94, 'LBBB': 81, 'RBBB': 50, 'PAC': 128, 'PVC': 154, 'ST': 111, 'STE': 59}

# paths to the data
FILES = {
    "partition": {
        "NAME": "ga_partition_64.json",
        "LOCATION": "../data/partition/btl",
    },
    "labels": {
        "NAME": "labels.json",
        "LOCATION": "../data/partition/btl",
    },
    "data": {
        "NAME": "",
        "LOCATION": "../../../data/cinc2020_b/",
        "FIELD_NAME": "filt",  # filt, raw for btl data
    },
}

# Maps labels to class indices
LABEL_MAPPING = {
    "Normal": 3,
    "AF": 0,
    "I-AVB": 1,
    "LBBB": 2,
    "RBBB": 6,
    "PAC": 4,
    "PVC": 5,
    "ST": 7,
    "STE": 8
}

# pre-processing pipeline
TRANSFORM = {
    "data": transforms.Compose([
        # transforms.ZScore(mean=, std=),
        transforms.HardClip(threshold=2500),
        transforms.RandomVerticalFlip(p=0.2),
    ]),
    "labels": transforms.Compose([
        transforms.OneHot(LABEL_MAPPING),
    ]),
}


# CUDA setting
CUDA = {
    "benchmark": True,
}

# PyTorch DataLoader batch parameters
BATCH = {
    "batch_size": 64,
    "shuffle": True,
    "num_workers": 6,
    "collate_fn": collate.PaddedCollate(dim=1, val=0),
}

# Learning parameters
LEARNING = {
    "max_epochs": 100,
    "lr": 0.1,
    "weight_decay": 0.0001,
}


