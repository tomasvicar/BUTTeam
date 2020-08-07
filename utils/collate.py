import torch
import numpy as np
import random
from config import Config


class PaddedCollate:
    """
    Create padded mini-batch of training samples along dimension dim
    """
    def __init__(self, dim=1, val=0):
        self.dim = dim
        self.padding_value = val
        self.num_channels = Config.INPUT_SIZE

    def pad_collate(self, batch):
        """
        Returns padded mini-batch
        :param batch: (list of tuples): tensor, label
        :return: padded_array - a tensor of all examples in 'batch' after padding
        labels - a LongTensor of all labels in batch
        sample_lengths – origin lengths of input data
        """
        batch_size = len(batch)
        # random_idx = random.shuffle(list(range(batch_size-1)))

        # find the longest sequence
        sample_lengths = [sample[0].shape[self.dim] for sample in batch]
        max_len = max(sample_lengths)

        # preallocate padded NumPy array
        shape = (batch_size, self.num_channels, max_len)
        padded_array = self.padding_value * np.ones(shape, dtype=np.float32)
        list_of_labels = []

        for idx, sample in enumerate(batch):
            padded_array[idx, :, :sample_lengths[idx]] = sample[0]
            list_of_labels.append(sample[1])

        # Pass to Torch Tensor
        padded_array = torch.from_numpy(padded_array).float()
        labels =torch.from_numpy(np.array(list_of_labels)).float() 
        sample_lengths = torch.LongTensor(sample_lengths)

        return padded_array, labels, sample_lengths

    def __call__(self, batch):
        return self.pad_collate(batch)
    