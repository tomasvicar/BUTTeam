import random
import numpy as np


__all__ = ["Compose", "HardClip", "ZScore", "RandomVerticalFlip", "OneHot",
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

    def __call__(self, data_sample):
        if self.transforms:
            for t in self.transforms:
                data_sample = t(data_sample)
        return data_sample


class HardClip(object):
    """Returns scaled and clipped data between range <-clipping_threshold:clipping_threshold>"""
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, sample):
        sample_mean = np.mean(sample, axis=1)
        sample = sample - sample_mean.reshape(-1, 1)
        sample[sample > self.threshold] = self.threshold
        sample[sample < -self.threshold] = -self.threshold
        sample = sample / self.threshold

        return sample


class ZScore(object):
    """Returns Z-score normalized data"""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample = sample - self.mean.reshape(-1, 1)
        sample= sample / self.std

        return sample


class RandomVerticalFlip(object):
    """Flip polarity of the given signal"""
    def __init__(self, p=0):
        """
        :param p (float): probability of each channel being flipped. Default = 0
        """
        self.p = p

    def __call__(self, sample):
        """
        :param sample (numpy array): multidimensional array
        :return: sample (numpy array)
        """
        for row in range(sample.shape[0]-1):
            if random.random() < self.p:
                sample[row, :] *= -1
        return sample


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
