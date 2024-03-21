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

def label_ratio_f(dataset, args):
    pos = 0
    neg = 0
    if not args["sex"]:
        for ecg, param in dataset:
            if param < 40:
                pos+=1
            else:
                neg+=1
        result = f"Positive LVEF(<40): {pos} and Negative LVEF: {neg}"
    else:
        for ecg, param in dataset:
            if param == 1:
                pos+=1
            else:
                neg+=1
        result = f"Gender Male: {pos} and Gender Female: {neg}"
        
    print(result)
    return result

def dataprep(args):
    dataDir = '/usr/sci/cibc/Maprodxn/ClinicalECGData/LVEFCohort/pythonData/'
    normEcgs = False

    print("Preparing Data For Classification Finetuning")
    with open('patient_splits/validation_patients.pkl', 'rb') as file:
        validation_patients = pickle.load(file)

    with open('patient_splits/pre_train_patients.pkl', 'rb') as file:
        pretrain_patients = pickle.load(file)
    
    num_classification_patients = len(pretrain_patients)
    finetuning_ratios = args["finetuning_ratios"]
    if args["sex"]:
        finetuning_ratios = [1]
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

        dataset = DataTools.ECGDatasetLoaderv2(baseDir=dataDir, patients=finetuning_patients.tolist(), normalize=normEcgs, sex=args["sex"])

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
    validation_dataset = DataTools.ECGDatasetLoaderv2(baseDir=dataDir, patients=validation_patients.tolist(), normalize=normEcgs, sex=args["sex"])
    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["workers"],
        pin_memory=True,
    )
    validation_ratio = label_ratio_f(validation_dataset, args)
    print(f"Preparing finetuning with {dataset_lengths} number of ECGs and with {len(validation_dataset)} validation ECGs")

    return train_loaders, val_loader, validation_ratio


def create_model(args):
    print("=> creating model '{}'".format("ECG Spatio Temporal"))
    model = Networks.ECG_SpatioTemporalNet(**parameters.spatioTemporalParams_v4, dim=1, mlp=args["mlp"])
    
    if args["sex"] or args["baseline"]:
        print("Preparing to Run Baseline without loading pre-trained weights")
        return model
    
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
    train_loaders, val_loader, validation_ratio = dataprep(args)

    logToWandB = args["logtowandb"]
    lossFun = T.loss_bce

    #Training
    print("Starting Training")

    date = datetime.datetime.now().date()


    for i, train_loader in enumerate(train_loaders):
        text = " for sex classification." if args["sex"] else ""
        print(f"Starting Finetuning with {len(train_loader.dataset)} patients{text}")

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

        freeze = "_freeze" if args["freeze_features"] else ""
        baseline = "_baseline" if args["baseline"] else ""
        label = "_sex" if args["sex"] else "_LVEF"
        

        project = f"MLECG_MoCO{label}_CLASSIFICATION{freeze}{baseline}_{date}"
        notes = f"Classification"
        config = dict(
            batch_size = args["batch_size"],
            learning_rate = args["lossParams"]["learningRate"],
            ngpus_per_node = tch.cuda.device_count(),
            epochs = args["finetuning_epochs"][i],
            mlp = args["mlp"],
            freeze_features = args["freeze_features"],
            lr_schedule = args["schedule"],
            baseline = args["baseline"],
            validation_ratio = validation_ratio,
            finetuning_examples = len(train_loader.dataset),
            validation_examples = len(val_loader.dataset),
            checkpoint = args["pretrained"]
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
                        numEpoch=args["finetuning_epochs"][i],
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






