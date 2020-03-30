from torch.utils import data
from read_file import read_data


class Dataset(data.Dataset):
    """
    PyTorch Dataset generator class
    Ref: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    """

    def __init__(self, list_of_ids, list_of_labels, data_path):
        """Initialization"""
        self.path = data_path
        self.labels = list_of_labels
        self.list_of_ids = list_of_ids

    def __len__(self):
        """Return total number of data samples"""
        return len(self.samples)

    def __getitem__(self, idx):
        """Generate data sample"""
        # Select sample
        file_name = self.list_of_ids[idx]

        # Read data and get label
        X = read_data(self.path, file_name)
        y = self.labels[id]

        return X, y


def main():
    return Dataset


if __name__ == "__main__":
    main()
