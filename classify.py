import os
import random
import warnings


import numpy as np
import torch as tch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import pickle


import DataTools
import main_moco
import main_lincls
import sex_classifcation
import loader


args = dict(
    seed = None,
    gpu = None,
    dist_url = "env://",
    world_size = -1,
    multiprocessing_distributed = True,
    dist_backend = "nccl",
    workers = 32,
    
    moco_dim = 128,
    moco_k = 65536,
    moco_m = 0.999,
    moco_t = 0.07,


    pretrain = True,
    start_epoch = 0,
    pretrain_epochs = 90,
    lr=0.03,
    momentum = 0.9,
    weight_decay = 1e-4,
    checkpoint_freq = 10,
    schedule = [30, 60],
    print_freq = 20,
    early_stop = 15,

    sex_classification = False,
    
    # finetuning_epochs = [40, 30, 30, 30, 30, 20],
    finetuning_epochs = [50, 40, 30],
    finetuning_ratios = [0.01, 0.05, 0.10],
    lossParams = dict(learningRate = 1e-4, threshold=40., type='binary cross entropy'),
    pretrained = '/usr/sci/cibc/ProjectsAndScratch/DeekshithMLECG/checkpoints/checkpoint_MagnitudeWarping/std_2_knots_2/checkpoint_best.pth.tar',
    slow_encoder = True,
    freeze_features = False,
    baseline = False,
        
    batch_size = 64,
    mlp = False,
    logtowandb = True,
    cos = False,

    
    checkpoint_dir = '/usr/sci/cibc/ProjectsAndScratch/DeekshithMLECG/checkpoints'

)

pretrained = [
    '/usr/sci/cibc/ProjectsAndScratch/DeekshithMLECG/checkpoints/checkpoint_RandomCropResize/random_crop_resize/checkpoint_best.pth.tar',
]

if __name__ == "__main__":
    for i in range(len(pretrained)):
        args['pretrained'] = pretrained[i]
        main_lincls.main_worker(args)
