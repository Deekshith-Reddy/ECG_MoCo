import random

import torch as tch
import torch.nn as nn


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