from torch.utils import data
from buttchery import loader


class Dataset(data.Dataset):
    """
    PyTorch Dataset generator class
    Ref: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    """

    def __init__(self, list_of_ids, list_of_labels, file_info, transform=None, encode=False):
        """Initialization"""
        self.file_info = file_info
        self.labels = list_of_labels
        self.list_of_ids = list_of_ids
        self.transform = transform
        self.encode = encode

    def __len__(self):
        """Return total number of data samples"""
        return len(self.list_of_ids)

    def __getitem__(self, idx):
        """Generate data sample"""
        # Select sample
        self.file_info["NAME"] = self.list_of_ids[idx]

        # Read data
        sample = loader.get_data(self.file_info)

        # Get label
        y = self.labels[self.file_info["NAME"]]

        # Transform sample
        if self.transform:
            sample = self.transform(sample)

        # Transform label
        if self.encode:
            y = self.encode(y)

        return sample, y


def main():
    return Dataset


if __name__ == "__main__":
    main()
