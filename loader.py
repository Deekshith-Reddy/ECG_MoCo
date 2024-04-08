import random

import torch as tch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.interpolate import CubicSpline

# Jittering
class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.3, mean=0):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
        self.mean = mean
    
    def forward(self ,x):
        # self.sig = random.uniform(self.std_dev[0], self.std_dev[1])
        self.sig = self.sigma
        self.noise = tch.randn(x.shape) * self.sig + self.mean
        return x + self.noise

# Blur  
class GaussianBlur(nn.Module):
    def __init__(self, sigma=0.6):
        super(GaussianBlur, self).__init__()
        self.sigma = sigma
        
    def forward(self, x):
        # self.sig = random.uniform(self.sigma[0], self.sigma[1])
        self.sig = self.sigma
        return tch.tensor(gaussian_filter(x, self.sig))


# Scaling
class Scaling(nn.Module):
    def __init__(self, sigma=0.2, mean=1):
        super(Scaling, self).__init__()
        self.sigma = sigma
        self.mean = mean
        
    
    def forward(self ,x):
        # self.std = random.uniform(self.sigma[0], self.sigma[1])
        self.std = self.sigma
        self.amp = tch.randn((1, 5000)) * self.std + self.mean
        return x * self.amp


# Magnitude Warping
class MagnitudeWarping(nn.Module):
    def __init__(self, knots=10, mean=1, sigma=0.2):
        super(MagnitudeWarping, self).__init__()
        self.knots = knots
        self.mean = mean
        self.sigma = sigma
    
    def forward(self, x):
        # self.std = random.uniform(self.sigma[0], self.sigma[1])
        self.std = self.sigma

        knot_indexes = np.linspace(0, x.shape[-1], self.knots, dtype=int)
        knot_values = np.random.normal(1, self.std, self.knots)

        spline = CubicSpline(knot_indexes, knot_values)
        self.warping = tch.tensor(spline(np.arange(x.shape[-1])))

        return x * self.warping


# Permutation
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

# Window Slicing
class RandomCropResize(tch.nn.Module):
    def __init__(self, crop_length_range=(4000, 5000), target_len=4500, interpolation='linear'):
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

# Time Warping
class TimeWarping(nn.Module):
    def __init__(self, mean=1.0, sigma=0.2, knots=4):
        super(TimeWarping, self).__init__()
        self.mean = mean
        self.sigma = sigma
        self.knots=knots
        

    def forward(self ,x):
        orig_steps = np.arange(x.shape[1])

        random_warps = np.random.normal(loc=self.mean, scale=self.sigma, size=(self.knots+2, x.shape[0]))
        warp_steps = (np.ones((x.shape[0], 1)) * (np.linspace(0, x.shape[1]-1., num=self.knots+2))).T

        ret = np.zeros_like(x)

        for lead in range(x.shape[0]):
            time_warp = CubicSpline(warp_steps[:,lead], warp_steps[:,lead] * random_warps[:,lead])(orig_steps)
            scale = (x.shape[1]-1)/time_warp[-1]
            ret[lead,:] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[1]-1,), x[lead,:]).T
        return tch.tensor(ret)

class BaselineWarping(nn.Module):
    def __init__(self, knots=10, mean=10, sigma=0.2):
        super(BaselineWarping, self).__init__()
        self.knots = knots
        self.mean = mean
        self.sigma = sigma
    
    def forward(self, x):
        # self.std = random.uniform(self.sigma[0], self.sigma[1])
        from scipy.interpolate import CubicSpline

        self.std = self.sigma

        knot_indexes = np.linspace(0, x.shape[-1], self.knots, dtype=int)
        knot_values = np.random.normal(1, self.std, self.knots)

        spline = CubicSpline(knot_indexes, knot_values)
        self.warping = tch.tensor(spline(np.arange(x.shape[-1])))

        return x + self.warping

# Window Warping
class WindowWarping(tch.nn.Module):
    def __init__(self, window_ratio=0.1, scales=[0.5, 2]):
        super(WindowWarping, self).__init__()
        self.window_ratio = window_ratio
        self.scales = scales

    
    def forward(self, x):
        warp_scale = np.random.choice(self.scales, 1)
        warp_size = np.ceil(self.window_ratio*x.shape[-1]).astype(int)
        window_steps = np.arange(warp_size)

        self.window_starts = np.random.randint(low=1, high=x.shape[-1]-warp_size-1, size=1).astype(int)[0]
        self.window_ends = (self.window_starts + warp_size).astype(int)

        ret = np.zeros_like(x)
        for lead in range(x.shape[0]):
            start_seg = x[lead, :self.window_starts]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scale)), window_steps, x[lead, self.window_starts:self.window_ends])
            end_seg = x[lead,self.window_ends:]
            warped = np.concatenate((start_seg, window_seg, end_seg))
            ret[lead, :] = np.interp(np.arange(x.shape[-1]), np.linspace(0, x.shape[-1]-1., num=warped.size), warped).T
        return tch.tensor(ret)



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






class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x).float()
        k = self.base_transform(x).float()
        return [q.unsqueeze(0), k.unsqueeze(0)]