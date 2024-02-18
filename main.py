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
    moco_dim = 128,
    moco_k = 65536,
    moco_m = 0.999,
    moco_t = 0.07,
    mlp = False,
    batch_size = 512,
    workers = 32,
    lr=0.03,
    momentum = 0.9,
    weight_decay = 1e-4,
    start_epoch = 0,
    lossParams = dict(learningRate = 1e-3, threshold=40., type='binary cross entropy'),
    pretrain_epochs = 150,
    finetuning_epochs = 20,
    cos = True,
    schedule = [120, 160],
    print_freq = 20,
    pretrained = 'checkpoints/checkpoint_0075.pth.tar',
    pretrain = True,
    freeze_features = True,
    logtowandb = True,

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
    pre_train_split_ratio = 0.8
    num_pre_train = int(pre_train_split_ratio * numPatients)
    num_classification = numPatients - num_pre_train

    random_seed_split = 1
    patientInds = list(range(numPatients))
    random.Random(random_seed_split).shuffle(patientInds)

    pre_train_patient_indices = patientInds[:num_pre_train]
    classification_patient_idices = patientInds[num_pre_train:num_pre_train + num_classification]

    pre_train_patients = patientIds[pre_train_patient_indices].squeeze()
    classification_patients = patientIds[classification_patient_idices].squeeze()

    with open('patient_splits/pre_train_patients.pkl', 'wb') as file:
        pickle.dump(pre_train_patients, file)
    with open('patient_splits/classification_patients.pkl', 'wb') as file:
        pickle.dump(classification_patients, file)

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
    
    date = datetime.datetime.now().date()

    if args["pretrain"]:
        project = f"MLECG_MoCO_LVEF_PRETRAIN_{date}"
        notes = "Pretrain"
        config = dict(
            batch_size = args["batch_size"],
            ngpus = ngpus_per_node,
            learning_rate = args["lr"],
            epochs = args["pretrain_epochs"]
        )
        networkLabel = "pre_train_ECG_SpatialTemporalNet"
    else:
        project = f"MLECG_MoCO_LVEF_CLASSIFICATION_{date}"
        notes = f"Classification"
        config = dict(
            batch_size = args["batch_size"],
            ngpus = ngpus_per_node,
            learning_rate = args["lossParams"]["learningRate"],
            epochs = args["finetuning_epochs"]
        )
        networkLabel = "Fine_tune_ECG_SpatialTemporalNet"

    # if args["logtowandb"]:
    #     wandbrun = wandb.init(
    #         project = project,
    #         notes=notes,
    #         tags=["training", "no artifact"],
    #         config=config,
    #         entity='deekshith',
    #         reinit=True,
    #         name=f"{networkLabel}_{datetime.datetime.now()}"
    # )
    
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
    # wandbrun.finish()


if __name__ == "__main__":
    main()







