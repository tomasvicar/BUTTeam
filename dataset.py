from torch.utils import data
from utils.datareader import DataReader
import utils.transforms
import numpy as np
import glob


class Dataset(data.Dataset):
    """
    PyTorch Dataset generator class
    Ref: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    """
    mean_age = 61.4

    def __init__(self, file_names, transform=None, encode=False):
        """Initialization"""
        self.snomed_table = DataReader.read_table(path="tables/")
        self.idx_mapping, self.label_mapping = DataReader.get_label_maps(path="tables/")
        self.file_names_list = file_names
        self.transform = transform
        self.encode = encode

    def __len__(self):
        """Return total number of data samples"""
        return len(self.file_names_list)

    def __getitem__(self, idx):
        """Generate data sample"""

        sample_file_name = self.file_names_list[idx]
        header_file_name = self.file_names_list[idx][:-3] + "hea"

        # Read data
        sample = DataReader.read_sample(sample_file_name)
        header = DataReader.read_header(header_file_name, self.snomed_table)

        sampling_frequency, resolution, age, sex, snomed_codes, labels = header

        if age is None:
            age = Dataset.mean_age

        # Transform sample
        if self.transform:
            sample = self.transform(sample, input_sampling=sampling_frequency)

        sample_length = sample.shape[1]

        # Transform label
        if self.encode:
            y = self.encode(snomed_codes, self.idx_mapping)

        return sample, y, sample_length, age, sex


def main():
    input_directory = r"E:\data\Physionet2020"
    file_list = glob.glob(input_directory + r"\**\*.mat", recursive=True)
    # file_list = [file_list[10330]]

    # Transform class composition
    t = transforms.Compose([
        transforms.Resample(output_sampling=500, gain=1),
        transforms.BaseLineFilter(window_size=1000),
        transforms.RandomAmplifier(p=0.3, max_multiplier=0.2),
    ])

    # Label encode composition
    encoder = transforms.SnomedToOneHot()

    # Generate Dataset
    training_set = Dataset(file_list, transform=t, encode=encoder)

    # Create generator
    training_generator = data.DataLoader(
        training_set,
    )

    # Train model
    for sample in training_generator:
        sample, y, sample_length, age, sex = sample

    # Convert one hot vector to Snomed codes
    decoder = transforms.OneHotToSnomed()
    idx_mapping, _ = DataReader.get_label_maps(path="")
    decoded_labels = decoder(np.array([0, 0, 1, 1, 0, 0]), idx_mapping)


if __name__ == "__main__":
    main()
