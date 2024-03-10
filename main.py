import os
import random
import warnings
import builtins
import math
import shutil
import time
import datetime

import numpy as np
import torch as tch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
import pickle
import wandb


import DataTools
import loader
import parameters
import Networks
import moco_builder
import main_moco
import main_lincls

os.environ["WANDB_API_KEY"] = "e56acefffc20a7f826010f436f392b067f4e0ae5"
device = tch.device("cuda" if tch.cuda.is_available() else "cpu")
gpuIds = list(range(tch.cuda.device_count()))
os.environ["WORLD_SIZE"] = "1"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'

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
    
    batch_size = 256,
    mlp = True,
    logtowandb = True,

    pretrain = False,
    pretrain_epochs = 90,
    lr=0.03,
    momentum = 0.9,
    weight_decay = 1e-4,
    cos = True,
    schedule = [30, 60],
    print_freq = 20,



    
    finetuning_epochs = [40, 30, 30, 30, 30, 20],
    lossParams = dict(learningRate = 1e-4, threshold=40., type='binary cross entropy'),
    start_epoch = 0,
    pretrained = 'checkpoints/checkpoint_0075.pth.tar',
    freeze_features = False,
    baseline = False,

)

def splitPatients(args):
    baseDir = ''
    dataDir = '/usr/sci/cibc/Maprodxn/ClinicalECGData/LVEFCohort/pythonData/'
    modelDir = ''
    normEcgs = False

    # Loading Data
    print('Finding Patients')
    allData = DataTools.PreTrainECGDatasetLoader(baseDir=dataDir, normalize=normEcgs)
    patientIds = np.array(allData.patients)
    numPatients = patientIds.shape[0]

    # Data
    pre_train_split_ratio = 0.9
    num_pre_train = int(pre_train_split_ratio * numPatients)
    num_validation = numPatients - num_pre_train

    random_seed_split = 1
    patientInds = list(range(numPatients))
    random.Random(random_seed_split).shuffle(patientInds)

    pre_train_patient_indices = patientInds[:num_pre_train]
    validation_patient_indices = patientInds[num_pre_train:num_pre_train + num_validation]

    pre_train_patients = patientIds[pre_train_patient_indices].squeeze()
    validation_patients = patientIds[validation_patient_indices].squeeze()

    with open('patient_splits/pre_train_patients.pkl', 'wb') as file:
        pickle.dump(pre_train_patients, file)
    with open('patient_splits/validation_patients.pkl', 'wb') as file:
        pickle.dump(validation_patients, file)
    print(f"Out of Total {numPatients} Splitting {len(pre_train_patients)} for pre-train and finetuning, {len(validation_patients)} for validation")

def main():
    if args["seed"] is not None:
        random.seed(args["seed"])
        tch.manual_seed(args["seed"])
        cudnn.deterministic = True
        warnings.warn(
            'You have chosen to seed training'
        )
    
    if args["gpu"] is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )
    
    if args["dist_url"] == "env://" and args["world_size"] == -1:
        args["world_size"] = int(os.environ["WORLD_SIZE"])

    args["distributed"] = args["world_size"] > 1 or args["multiprocessing_distributed"]

    splitPatients(args)

    ngpus_per_node = tch.cuda.device_count()
    
    
    if args["pretrain"]:
        if args["multiprocessing_distributed"]:
            args["world_size"] = ngpus_per_node * args["world_size"]
            mp.spawn(main_moco.main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        else:
            # Simply call main_worker function
            main_moco.main_worker(args.gpu, ngpus_per_node, args)
    else:
        main_lincls.main_worker(args)

    print("DUNDANADUN")


if __name__ == "__main__":
    main()







