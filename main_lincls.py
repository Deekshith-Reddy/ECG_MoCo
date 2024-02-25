import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import datetime

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
import wandb

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
    with open('patient_splits/validation_patients.pkl', 'rb') as file:
        validation_patients = pickle.load(file)

    with open('patient_splits/pre_train_patients.pkl', 'rb') as file:
        pretrain_patients = pickle.load(file)
    
    num_classification_patients = len(pretrain_patients)
    finetuning_ratios = [1.0, 0.75, 0.50, 0.10, 0.05, 0.01]
    num_finetuning = [int(num_classification_patients * r) for r in finetuning_ratios]
    print(f"Num of classifcation patients is {num_classification_patients} Patients split as {num_finetuning}")
    
    random_seed_split = 1
    patientInds = list(range(num_classification_patients))
    random.Random(random_seed_split).shuffle(patientInds)

    train_loaders = []
    dataset_lengths = []
    for i in num_finetuning:
        finetuning_patient_indices = patientInds[:i]

        finetuning_patients = pretrain_patients[finetuning_patient_indices].squeeze()

        dataset = DataTools.ECGDatasetLoader(baseDir=dataDir, patients=finetuning_patients.tolist(), normalize=normEcgs)

        loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args["workers"],
        pin_memory=True,
        )

        train_loaders.append(loader)
        dataset_lengths.append(len(dataset))
    
    validation_patients = validation_patients
    validation_dataset = DataTools.ECGDatasetLoader(baseDir=dataDir, patients=validation_patients.tolist(), normalize=normEcgs)
    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["workers"],
        pin_memory=True,
    )

    print(f"Preparing finetuning with {dataset_lengths} number of ECGs and with {len(validation_dataset)} validation ECGs")

    return train_loaders, val_loader

def create_model(args):
    print("=> creating model '{}'".format("ECG Spatio Temporal"))
    model = Networks.ECG_SpatioTemporalNet(**parameters.spatioTemporalParams_v4, dim=1, mlp=args["mlp"])
    for name, param in model.named_parameters():
        if not name.startswith("finalLayer."):
            param.requires_grad = not args["freeze_features"]
    
    if args["pretrained"]:
        if os.path.isfile(args["pretrained"]):
            print("=> loading checkpoint '{}'".format(args["pretrained"]))
            checkpoint = tch.load(args["pretrained"], map_location="cpu")

            state_dict = checkpoint["state_dict"]
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith("module.encoder_q") and not k.startswith(
                    "module.encoder_q.finalLayer."
                ):
                    # remove prefix
                    state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {'finalLayer.2.bias', 'finalLayer.2.weight'} if not args["mlp"] else {
                 'finalLayer.2.weight', 'finalLayer.2.bias', 
                 'finalLayer.4.weight', 'finalLayer.4.bias', 
                 'finalLayer.6.weight', 'finalLayer.6.bias', 
                 'finalLayer.8.weight', 'finalLayer.8.bias'

            }

            print("=> loaded pre-trained model '{}'".format(args["pretrained"]))
        else:
            print("=> No checkpoint found at '{}'".format(args["pretrained"]))

    return model


def main_worker(args):

     
    
    # Data Loading
    train_loaders, val_loader = dataprep(args)

    logToWandB = args["logtowandb"]
    lossFun = T.loss_bce

    #Training
    print("Starting Training")

    date = datetime.datetime.now().date()

    for train_loader in train_loaders:
        print(f"Starting Finetuning with {len(train_loader.dataset)} patients")

        model = create_model(args)   
    
        model = tch.nn.DataParallel(model, device_ids=gpuIds)   
        print(model)
        model.to(device)

        lossParams = args["lossParams"]


        optimizer = tch.optim.SGD(
            model.parameters(),
            args["lr"],
            momentum=args["momentum"],
            weight_decay=args["weight_decay"]
        )
        
        optimizer1 = tch.optim.Adam(model.parameters(), lr=lossParams['learningRate'])
    

        project = f"MLECG_MoCO_LVEF_CLASSIFICATION_{date}"
        notes = f"Classification"
        config = dict(
            batch_size = args["batch_size"],
            learning_rate = args["lossParams"]["learningRate"],
            ngpus_per_node = tch.cuda.device_count(),
            epochs = args["finetuning_epochs"],
            mlp = args["mlp"],
            freeze_features = args["freeze_features"],
            lr_schedule = args["schedule"],
            finetuning_examples = len(train_loader.dataset)
        )
        networkLabel = "Fine_tune_ECG_SpatialTemporalNet"

        if logToWandB:
            wandbrun = wandb.init(
                project = project,
                notes=notes,
                tags=["training", "no artifact"],
                config=config,
                entity='deekshith',
                reinit=True,
                name=f"{networkLabel}_{len(train_loader.dataset)}_{datetime.datetime.now()}"
            )

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
        if logToWandB:
            wandbrun.finish()
    





