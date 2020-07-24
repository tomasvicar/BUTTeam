import random
import os
import numpy as np
import scipy.io as io
from scipy import signal
import torch
from utils.datareader import DataReader

__all__ = ["Compose", "HardClip", "ZScore", "RandomShift", "RandomStretch", "RandomAmplifier", "RandomVerticalFlip",
           "Resample", "OneHot",
           ]


class Compose(object):
    """Composes several transforms together.
    Example:
        transforms.Compose([
            transforms.HardClip(10),
            transforms.ToTensor(),
            ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data_sample, **kwargs):
        if self.transforms:
            for t in self.transforms:
                data_sample = t(data_sample, **kwargs)
        return data_sample


class HardClip(object):
    """Returns scaled and clipped data between range <-clipping_threshold:clipping_threshold>"""
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, sample, **kwargs):
        sample_mean = np.mean(sample, axis=1)
        sample = sample - sample_mean.reshape(-1, 1)
        sample[sample > self.threshold] = self.threshold
        sample[sample < -self.threshold] = -self.threshold
        sample = sample / self.threshold

        return sample


class ZScore:
    """Returns Z-score normalized data"""
    def __init__(self, mean=0, std=1000):
        self.mean = mean
        self.std = std

    def __call__(self, sample, **kwargs):
        sample = sample - np.array(self.mean).reshape(-1, 1)
        sample = sample / self.std

        return sample


class RandomShift:
    """
        Class randomly shifts signal within temporal dimension
    """
    def __init__(self, p=0):
        self.probability = p

    def __call__(self, sample, **kwargs):
        self.sample_length = sample.shape[1]
        self.sample_channels = sample.shape[0]

        if random.random() > self.probability:
            shift = torch.randint(self.sample_length, (1, 1)).view(-1).numpy()
            return np.roll(sample, shift, axis=1)


class RandomStretch:
    """
    Class randomly stretches temporal dimension of signal
    """
    def __init__(self, p=0, max_stretch=0.1):
        self.probability = p
        self.max_stretch = max_stretch

    def __call__(self, sample, **kwargs):
        self.sample_length = sample.shape[1]
        self.sample_channels = sample.shape[0]

        if random.random() > self.probability:
            relative_change = 1 + torch.rand(1).numpy()[0] * 2 * self.max_stretch - self.max_stretch
            new_len = int(relative_change * self.sample_length)

            stretched_sample = np.zeros((self.sample_channels, new_len))
            for channel_idx in range(self.sample_channels):
                stretched_sample[channel_idx, :] = np.interp(np.linspace(0, self.sample_length - 1, new_len),
                                                             np.linspace(0, self.sample_length - 1, self.sample_length),
                                                             sample[channel_idx, :])
            return stretched_sample


class RandomAmplifier:
    """
    Class randomly amplifies signal
    """
    def __init__(self, p=0, max_multiplier=0.2):
        self.probability = p
        self.max_multiplier = max_multiplier

    def __call__(self, sample, **kwargs):
        self.sample_length = sample.shape[1]
        self.sample_channels = sample.shape[0]

        if random.random() > self.probability:
            for channel_idx in range(sample.shape[0]):
                multiplier = 1 + random.random() * 2 * self.max_multiplier - self.max_multiplier
                sample[channel_idx, :] = sample[channel_idx, :] * multiplier

        return sample


class RandomVerticalFlip(object):
    """Flip polarity of the given signal"""
    """Should be only I and aVL"""
    def __init__(self, p=0):
        self.probability = p

    def __call__(self, sample, **kwargs):
        """
        :param sample (numpy array): multidimensional array
        :return: sample (numpy array)
        """
        for row in range(sample.shape[0]-1):
            if random.random() < self.p:
                sample[row, :] *= -1
        return sample


class Resample:
    def __init__(self, output_sampling=500, gain=1):
        self.gain = gain
        self.output_sampling = int(output_sampling)

    def __call__(self, sample, input_sampling):
        # if current_gain != self.gain:
        #     current_gain = current_gain / self.gain

        # Rescale data
        self.sample = sample
        self.input_sampling = int(input_sampling)

        # Resample data
        if self.input_sampling < self.output_sampling:
            self._upsample(self.output_sampling / self.input_sampling)
        elif self.input_sampling > self.output_sampling:
            lowest_multiple = np.lcm(self.input_sampling, self.output_sampling)
            self._downsample(lowest_multiple // self.input_sampling, lowest_multiple // self.output_sampling)
        else:
            return self.sample

        return self.resampled_sample

    def _upsample(self, up_factor, down_factor=1):
        new_length = int(up_factor / down_factor * self.sample.shape[1])
        self.resampled_sample = np.zeros((self.sample.shape[0], new_length))

        for channel_idx in range(self.sample.shape[0]):
            self.resampled_sample[channel_idx, :] = np.interp(np.linspace(0, self.sample.shape[1] - 1, new_length),
                                                              np.linspace(0, self.sample.shape[1] - 1, self.sample.shape[1]),
                                                              self.sample[channel_idx, :])

    def _downsample(self, up_factor, down_factor=1):
        new_length = int(up_factor / down_factor * self.sample.shape[1])
        self.resampled_sample = np.zeros((self.sample.shape[0], new_length))

        for channel_idx in range(self.sample.shape[0]):
            self.resampled_sample[channel_idx, :] = signal.resample_poly(self.sample[channel_idx, :],
                                                                         up_factor,
                                                                         down_factor,
                                                                         )


class BaseLineFilter:
    def __init__(self, window_size=1000):
        self.window_size = window_size

    def __call__(self, sample, **kwargs):
        for channel_idx in range(sample.shape[0]):
            running_mean = BaseLineFilter._running_mean(sample[channel_idx], self.window_size)
            sample[channel_idx] = sample[channel_idx] - running_mean
        return sample

    @staticmethod
    def _running_mean(sample, window_size):
        window = signal.windows.blackman(window_size)
        window = window / np.sum(window)
        return signal.fftconvolve(sample, window, mode="same")


class OneHot(object):
    """Returns one hot encoded labels"""
    def __init__(self, mapping):
        self.mapping = mapping
        self.length = len(mapping)

    def __call__(self, labels):
        encoded_labels = np.zeros(self.length)
        for name in labels:
            encoded_labels[self.mapping[name]] = True

        return encoded_labels


class SnomedToOneHot(object):
    """Returns one hot encoded labels"""
    def __init__(self):
        pass

    def __call__(self, snomed_codes, mapping):
        encoded_labels = np.zeros(len(mapping)).astype(np.float32)
        for code in snomed_codes:
            if code not in mapping:
                continue
            else:
                encoded_labels[mapping[code]] = 1.0

        return encoded_labels


class OneHotToSnomed(object):
    """Returns one hot encoded labels"""
    def __init__(self):
        pass

    def __call__(self, one_hot_vector, mapping):
        inverse_mapping = {value: key for key, value in mapping.items()}
        return [inverse_mapping[idx] for idx, value in enumerate(one_hot_vector) if value > 0]


def main():
    file_path = "E:\\data\\Physionet2020\\Training_StPetersburg\\I0043.mat"
    sample = io.loadmat(os.path.join(file_path))
    sample = sample["val"]

    # Transform class composition
    t = Compose([
        Resample(output_sampling=500, gain=1),
        BaseLineFilter(window_size=1000),
        RandomAmplifier(p=0.3, max_multiplier=0.2),
    ])

    # One hot vector to Snomed code
    decoder = OneHotToSnomed()
    idx_mapping, label_mapping = DataReader.get_label_maps(path="")
    decoded_labels = decoder(np.array([0, 0, 1, 1, 0, 0]), idx_mapping)

    # Sample transforms
    x = t(sample, input_sampling=1000)


if __name__ == "__main__":
    main()
