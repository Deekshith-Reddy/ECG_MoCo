import random
import datetime


import torch as tch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import pickle
import wandb

import Networks
import parameters
import DataTools
import Training as T

device = tch.device("cuda" if tch.cuda.is_available() else "cpu")
gpuIds = list(range(tch.cuda.device_count()))

def dataprep(args):
    dataDir = '/usr/sci/cibc/Maprodxn/ClinicalECGData/LVEFCohort/pythonData/'
    normEcgs = False

    print("Preparing Data for Classifcation Finetuning")
    with open('patient_splits/validation_patients.pkl', 'rb') as file:
        validation_patients = pickle.load(file)

    with open('patient_splits/pre_train_patients.pkl', 'rb') as file:
        classification_patients = pickle.load(file)
    
    num_classification_patients = len(classification_patients)
    num_validation_patients = len(validation_patients)
    
    print(f"Num of patients used for SEX Classification is {num_classification_patients} and Validation is {num_validation_patients}")

    random_seed = 42
    patientInds = list(range(num_classification_patients))
    random.Random(random_seed).shuffle(patientInds)

    classification_patients = classification_patients[patientInds]
    
    train_dataset = DataTools.ECG_Sex_DatasetLoader(baseDir=dataDir, patients=classification_patients.tolist(), normalize=normEcgs)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args["workers"],
        pin_memory=True

    )
    
    validation_dataset = DataTools.ECG_Sex_DatasetLoader(baseDir=dataDir, patients=validation_patients.tolist(), normalize=normEcgs)
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["workers"],
        pin_memory=True
    )

    print(f"Preparing Finetuning with {len(train_dataset)} ECGs and Validation with {len(validation_dataset)} ECGs for SEX Classification")

    return train_loader, validation_loader

def main_worker(args):

    # Data Loading
    train_loader, validation_loader = dataprep(args)

    logToWandB = args["logtowandb"]
    lossFun = T.loss_bce

    #Training
    print("Starting Training")
    date = datetime.datetime.now().date()

    print("=> creating model '{}'".format("ECG Spatio Temporal"))
    model = Networks.ECG_SpatioTemporalNet(**parameters.spatioTemporalParams_v4, dim=1, mlp=args["mlp"])

    model = tch.nn.DataParallel(model, device_ids=gpuIds)
    print(model)
    model.to(device)

    lossParams = args["lossParams"]

    optimizer = tch.optim.Adam(model.parameters(), lr=lossParams['learningRate'])


    project = f"MLECG_SEX_Classification"
    notes = f"Classifcation for gender identification using all training examples of size {len(train_loader.dataset)} and validation on size {len(validation_loader.dataset)}"
    
    config = dict (
        batch_size = args["batch_size"],
        learning_rate = args["lossParams"]["learningRate"],
        epochs = args["num_epochs"],
        mlp = args["mlp"],
        lr_schedule = args["schedule"],
        training_size = len(train_loader.dataset),
        validation_size = len(validation_loader.dataset),
    )

    networkLabel = "ECG_SpatialTemporalNet"

    if logToWandB:
        wandbrun = wandb.init(
            project = project,
            notes = notes,
            tags = [str(date)],
            config = config,
            entity = 'deekshith',
            reinit = True,
            name = f"{networkLabel}_SEX_Classification_{datetime.datetime.now()}"
        )

    T.trainNetwork (
        network=model,
        trainDataLoader=train_loader,
        testDataLoader=validation_loader,
        numEpoch=args['num_epochs'],
        optimizer=optimizer,
        lossFun=lossFun,
        lossParams=lossParams,
        modelSaveDir='models/sex_classification/',
        label=networkLabel,
        args=args,
        logToWandB=logToWandB,
        problemType='Binary'


    )

    if logToWandB:
        wandbrun.finish()



