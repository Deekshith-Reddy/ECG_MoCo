import argparse
import builtins
import os
import random
import shutil
import time
import warnings

import torch as tch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import pickle

import Networks
import parameters
import DataTools
import Training as T

best_acc1 = 0
device = tch.device("cuda" if tch.cuda.is_available() else "cpu")
gpuIds = list(range(tch.cuda.device_count()))

def dataprep(args):
    dataDir = '/usr/sci/cibc/Maprodxn/ClinicalECGData/LVEFCohort/pythonData/'
    normEcgs = False

    print("Preparing Data For Classification Finetuning")
    with open('patient_splits/classification_patients.pkl', 'rb') as file:
        classification_patients = pickle.load(file)
    
    num_classification_patients = len(classification_patients)
    finetuning_ratio = 0.5
    num_finetuning = int(num_classification_patients * finetuning_ratio)
    num_validation = num_classification_patients - num_finetuning

    random_seed_split = 1
    patientInds = list(range(num_classification_patients))
    random.Random(random_seed_split).shuffle(patientInds)

    finetuning_patient_indices = patientInds[:num_finetuning]
    validation_patient_indices = patientInds[num_finetuning:num_finetuning + num_validation]

    finetuning_patients = classification_patients[finetuning_patient_indices].squeeze()
    validation_patients = classification_patients[validation_patient_indices]

    train_dataset = DataTools.ECGDatasetLoader(baseDir=dataDir, patients=finetuning_patients.tolist(), normalize=normEcgs)
    validation_dataset = DataTools.ECGDatasetLoader(baseDir=dataDir, patients=validation_patients.tolist(), normalize=normEcgs)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args["workers"],
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["workers"],
        pin_memory=True,
    )

    return train_loader, val_loader

def main_worker(args):
    
    print("=> creating model '{}'".format("ECG Spatio Temporal"))
    model = Networks.ECG_SpatioTemporalNet(**parameters.spatioTemporalParams_v4, dim=1)
    for name, param in model.named_parameters():
        if name not in ["finalLayer.2.weight", "finalLayer.2.bas"]:
            param.requires_grad = not args["freeze_features"]
    
    if args["pretrained"]:
        if os.path.isfile(args["pretrained"]):
            print("=> loading checkpoint '{}'".format(args["pretrained"]))
            checkpoint = tch.load(args["pretrained"], map_location="cpu")

            state_dict = checkpoint["state_dict"]
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith("module.encoder_q") and not k.startswith(
                    "module.encoder_q.finalLayer.2."
                ):
                    # remove prefix
                    state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {'finalLayer.2.bias', 'finalLayer.2.weight'}

            print("=> loaded pre-trained model '{}'".format(args["pretrained"]))
        else:
            print("=> No checkpoint found at '{}'".format(args["pretrained"]))

    
    model = tch.nn.DataParallel(model, device_ids=gpuIds)   
    print(model)
    model.to(device) 
    
    # Data Loading
    train_loader, val_loader = dataprep(args)

    lossParams = dict(learningRate = 1e-3, threshold=40., type='binary cross entropy')


    optimizer = tch.optim.SGD(
        model.parameters(),
        args["lr"],
        momentum=args["momentum"],
        weight_decay=args["weight_decay"]
    )
    
    optimizer1 = tch.optim.Adam(model.parameters(), lr=lossParams['learningRate'])
    
    logToWandB = False
    lossFun = T.loss_bce

    #Training
    print("Starting Training")
    T.trainNetwork(
                    network=model,
                    trainDataLoader=train_loader,
                    testDataLoader=val_loader,
                    numEpoch=args["finetuning_epochs"],
                    optimizer=optimizer1,
                    lossFun=lossFun,
                    lossParams=lossParams,
                    modelSaveDir='models/',
                    label='networkLabel',
                    args=args,
                    logToWandB=logToWandB,
                    problemType='Binary'
                )
    





