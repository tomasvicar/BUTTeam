import random
import os
import numpy as np
import scipy.io as io
from scipy import signal
from scipy.signal import firwin,filtfilt
import torch
from utils.datareader import DataReader

__all__ = ["Compose", "HardClip", "ZScore", "RandomShift", "RandomStretch", "RandomAmplifier", "RandomLeadSwitch",
           "Resample", "BaseLineFilter", "OneHot", "SnomedToOneHot", "OneHotToSnomed", "AddEmgNoise"
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

        if random.random() < self.probability:
           
            shift = torch.randint(self.sample_length, (1, 1)).view(-1).numpy()
            
            sample=np.roll(sample, shift, axis=1)
            
        return sample


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

        if random.random() < self.probability:
            relative_change = 1 + torch.rand(1).numpy()[0] * 2 * self.max_stretch - self.max_stretch
            if relative_change<1:
                relative_change=1/(1-relative_change+1)
            
            
            new_len = int(relative_change * self.sample_length)

            stretched_sample = np.zeros((self.sample_channels, new_len))
            for channel_idx in range(self.sample_channels):
                stretched_sample[channel_idx, :] = np.interp(np.linspace(0, self.sample_length - 1, new_len),
                                                             np.linspace(0, self.sample_length - 1, self.sample_length),
                                                             sample[channel_idx, :])
                
            sample=stretched_sample
        return sample


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

        if random.random() < self.probability:
            for channel_idx in range(sample.shape[0]):
                multiplier = 1 + random.random() * 2 * self.max_multiplier - self.max_multiplier
                
                ##mutliply by 2 is same as equvalent to multiply by 0.5 not 0!
                if multiplier<1:
                    multiplier=1/(1-multiplier+1)
                    
                sample[channel_idx, :] = sample[channel_idx, :] * multiplier

        return sample


class RandomLeadSwitch(object):
    """Simulates reversal of ecg leads"""
    """Should be only I and aVL"""

    def __init__(self, p=0.05):
        self.probability = p
        self.reversal_type = ["LA_LR", "LA_LL", "RA_LL", "PRECORDIAL"]
        self.weights = [3, 1, 1, 2]
        self.precordial_pairs = [("V1", "V2"), ("V2", "V3"), ("V3", "V4"), ("V4", "V5"), ("V5", "V6")]
        self.lead_map = dict(zip(
            ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
            range(0, 12))
        )

    def __call__(self, sample, **kwargs):
        """
        :param sample (numpy array): multidimensional array
        :return: sample (numpy array)
        """
        self.sample = sample

        if random.random() < self.probability:
            selected_type = random.choices(self.reversal_type, weights=self.weights, k=1)[0]
            if selected_type == "LA_LR":
                self.invert_channel("I")
                self.switch_channel(["II", "III"])
                self.switch_channel(["aVL", "aVR"])
                return self.sample

            if selected_type == "LA_LL":
                self.invert_channel("III")
                self.switch_channel(["I", "II"])
                self.switch_channel(["aVL", "aVF"])
                return self.sample

            if selected_type == "RA_LL":
                self.invert_channel("I")
                self.invert_channel("II")
                self.invert_channel("III")
                self.switch_channel(["I", "III"])
                self.switch_channel(["aVR", "aVF"])
                return self.sample

            if selected_type == "PRECORDIAL":
                self.switch_channel(random.choices(self.precordial_pairs, k=1)[0])
                return self.sample
        else:
            return self.sample

    def invert_channel(self, channel_name):
        self.sample[self.lead_map[channel_name], :] *= -1

    def switch_channel(self, channel_names):
        self.sample[[self.lead_map[channel_names[0]], self.lead_map[channel_names[1]]], :] = \
            self.sample[[self.lead_map[channel_names[1]], self.lead_map[channel_names[0]]], :]


class Resample:
    def __init__(self, output_sampling=500):
        self.output_sampling = int(output_sampling)

    def __call__(self, sample, input_sampling,gain):
        
        sample=sample.astype(np.float32)
        for k in range(sample.shape[0]):
            sample[k,:]=sample[k,:]*gain[k]

        # Rescale data
        self.sample = sample
        self.input_sampling = int(input_sampling)
        
        factor=self.output_sampling / self.input_sampling
        
        len_old=self.sample.shape[1]
        num_of_leads=self.sample.shape[0]
        

        new_length = int(factor * len_old)
        resampled_sample = np.zeros((num_of_leads, new_length))

        for channel_idx in range(num_of_leads):
            tmp=self.sample[channel_idx, :]
            
            ### antialias
            if factor<1:
                q=1/factor
                
                half_len = 10 * q  
                n = 2 * half_len
                b, a = firwin(int(n)+1, 1./q, window='hamming'), 1.
                tmp = filtfilt(b, a, tmp)
            
            
            l1=np.linspace(0,len_old - 1, new_length)
            l2=np.linspace(0,len_old - 1, len_old)
            tmp= np.interp(l1,l2,tmp)
            resampled_sample[channel_idx, :] = tmp

        return resampled_sample


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


class AddEmgNoise(object):
    """Returns one hot encoded labels"""
    def __init__(self, file_name, path="", p=0.5, min_length=500, max_length=2000, magnitude_coeff=10):
        self.emg_noise = AddEmgNoise._get_file(os.path.join(path, file_name))
        self.emg_length = len(self.emg_noise)
        self.probability = p
        self.min_length = min_length
        self.max_length = max_length
        self.magnitude_coeff = magnitude_coeff

    def __call__(self, sample):

        # repeatedly add EMG artifact into ECG record
        for lead_idx in range(sample.shape[0]):
            if random.random() < self.probability:
                continue

            lead_std = np.std(sample[lead_idx])
            samples_to_distort = int(sample.shape[1] * (random.random() + 1) / 3)
            distorted_samples = set()

            while len(distorted_samples) < samples_to_distort:
                noise_length = random.randint(self.min_length, self.max_length)
                _from = random.randint(0, sample.shape[1] - noise_length)
                _to = _from + noise_length
                noise = self.magnitude_coeff * random.random() * lead_std * self.generate_noise(noise_length)

                # Add noise to ecg lead
                sample[lead_idx, _from:_to] = sample[lead_idx, _from:_to] + noise

                # Update distorted indices
                distorted_samples.update(list(range(_from, _to)))

        return sample

def main():
    file_path = "E:\\data\\Physionet2020\\Training_StPetersburg\\I0043.mat"
    sample = io.loadmat(os.path.join(file_path))
    sample = sample["val"]

    # Transform class composition
    t = Compose([
        Resample(output_sampling=500, gain=1),
        BaseLineFilter(window_size=1000),
        RandomAmplifier(p=0.3, max_multiplier=0.2),
        RandomLeadSwitch(p=0.05),
        AddEmgNoise(file_name="emg_raw.txt", path="", p=0.3, min_length=500, max_length=2000, magnitude_coeff=10),
    ])

    # One hot vector to Snomed code
    decoder = OneHotToSnomed()
    idx_mapping, label_mapping = DataReader.get_label_maps(path="")
    decoded_labels = decoder(np.array([0, 0, 1, 1, 0, 0]), idx_mapping)

    # Sample transforms
    x = t(sample, input_sampling=1000)


if __name__ == "__main__":
    main()
