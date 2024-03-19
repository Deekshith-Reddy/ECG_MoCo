import random

import torch as tch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d


class GuassianBlur(nn.Module):
    def __init__(self, std_dev=(0.1, 0.25), mean=0):
        super(GuassianBlur, self).__init__()
        self.std_dev = std_dev
        self.mean = mean
    
    def forward(self ,x):
        sig = random.uniform(0.1, 0.25)
        noise = tch.randn(x.shape) * sig + self.mean
        return x + noise

class Permutation(nn.Module):
    def __init__(self, p=(4, 10), leadbylead=True):
        super(Permutation, self).__init__()
        self.p = p
        self.leadbylead = leadbylead
    
    def forward(self ,x):
        subs = random.randint(self.p[0], self.p[1])
        total_time_points = x.shape[1]
        subsection_size = total_time_points // subs

        permuted_signal = tch.zeros_like(x)

        if self.leadbylead:
            for lead in range(x.shape[0]):
                subsections = [x[lead, i:i+subsection_size] for i in range(0, total_time_points, subsection_size)]
                random.shuffle(subsections)
                permuted_signal[lead] = tch.concat(subsections, dim=0)
        else:
            subsections = [x[:,i:i+subsection_size] for i in range(0, total_time_points, subsection_size)]
            random.shuffle(subsections)
            permuted_signal = tch.concat(subsections, dim=-1)


        return permuted_signal

class ZeroMask(nn.Module):
    def __init__(self, r=0.4, leadbylead=True):
        super(ZeroMask, self).__init__()
        self.r = r
        self.leadbylead = leadbylead
    
    def forward(self ,x):
        leads, L = x.shape
        num_samples_mask = int(self.r * L)

        X_masked = x.clone()
        
        if self.leadbylead:
            for lead in range(leads):
                start_point = tch.randint(0, L - num_samples_mask + 1, (1,)).item()
                X_masked[lead, start_point:start_point + num_samples_mask] = 0
        else:
            start_point = tch.randint(0, L - num_samples_mask + 1, (1,)).item()
            X_masked[:, start_point:start_point + num_samples_mask] = 0

        return X_masked

class RandomCropResize(tch.nn.Module):
    def __init__(self, crop_length_range=(2500, 3500), target_len=3000, interpolation='linear'):
        super(RandomCropResize, self).__init__()
        self.crop_length_range = crop_length_range
        self.target_len = target_len
        self.interpolation = interpolation

    def forward(self, x):
        crop_length = tch.randint(self.crop_length_range[0], self.crop_length_range[1], (1, ))[0]
        if crop_length == 5000:
            start_point = 0
        else:
            start_point = tch.randint(0, x.shape[1] - crop_length, size=(1,)).item()
        
        cropped_signal = x[:, start_point:start_point + crop_length]
        num_leads, curr_len = cropped_signal.shape
        resized_signal = tch.zeros((num_leads, self.target_len))

        x_original = tch.linspace(0, 1, curr_len)
        x_target = tch.linspace(0, 1, self.target_len)

        for i in range(num_leads):
            f = interp1d(x_original, cropped_signal[i], kind=self.interpolation)
            resized_signal[i] = tch.tensor(f(x_target))
        return resized_signal


class Scaling(nn.Module):
    def __init__(self, p):
        print(p)
    
    def forward(self ,x):
        return x

class TimeWarping(nn.Module):
    def __init__(self, p):
        print(p)

    def forward(self ,x):
        return x


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q.unsqueeze(0), k.unsqueeze(0)]