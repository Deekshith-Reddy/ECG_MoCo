import os
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

    print("Preparing Data For Classification Finetuning")
    with open('patient_splits/validation_patients.pkl', 'rb') as file:
        validation_patients = pickle.load(file)

    with open('patient_splits/pre_train_patients.pkl', 'rb') as file:
        pretrain_patients = pickle.load(file)
    
    num_classification_patients = len(pretrain_patients)
    finetuning_ratios = args["finetuning_ratios"]
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
    
    if args["baseline"]:
        print("Preparing to Run Baseline without loading pre-trained weights")
        return model
    
    for name, param in model.named_parameters():
        if not name.startswith("finalLayer."):
            param.requires_grad = not args["freeze_features"]
    
    if args["pretrained"]:
        if os.path.isfile(args["pretrained"]):
            print("=> loading checkpoint '{}'".format(args["pretrained"]))
            checkpoint = tch.load(args["pretrained"], map_location="cpu")
            print(f"Loaded checkpoint @Epoch {checkpoint['epoch']}")
            state_dict = checkpoint["state_dict"]
            if args["pretrained"].find("sex") == -1:
                for k in list(state_dict.keys()):
                            # retain only encoder_q up to before the embedding layer
                            if k.startswith("module.encoder_q") and not k.startswith(
                                "module.encoder_q.finalLayer."
                            ):
                                # remove prefix
                                state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
                            # delete renamed or unused k
                            del state_dict[k]
            else:
                for k in list(state_dict.keys()):
                            # retain only encoder_q up to before the embedding layer
                            if k.startswith("module.") and not k.startswith(
                                "module.finalLayer."
                            ):
                                # remove prefix
                                state_dict[k[len("module.") :]] = state_dict[k]
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

    for i, train_loader in enumerate(train_loaders):
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

        params = [{'params':getattr(model,i).parameters(), 'lr': 1e-5} if i.find("finalLayer")==-1 else {'params':getattr(model,i).parameters(), 'lr': 1e-3} for i,x in model.named_children()]

        if args["slow_encoder"]:
            optimizer1 = tch.optim.Adam(params)
            print("Using the slow encoder with learning rates of the encoder being 1e-5 and the finalLayer being 1e-3")
        else:
            optimizer1 = tch.optim.Adam(model.parameters(), lr=lossParams['learningRate'])

        freeze = "_freeze" if args["freeze_features"] else ""
        baseline = "_baseline" if args["baseline"]  else ""
        MoCo = "_MoCo" if args["pretrained"].find("sex") == -1 else ""
        on_sex = "" if args["pretrained"].find("sex") == -1 and not args["baseline"] else "_ON_SEX"
    

        project = f"MLECG_{MoCo}_LVEF_CLASSIFICATION{freeze}{baseline}{on_sex}_{date}"
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
                name=f"{networkLabel}_{len(train_loader.dataset)}_{datetime.datetime.now()}",
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
                        label=networkLabel,
                        args=args,
                        logToWandB=logToWandB,
                        problemType='Binary'
                    )
        if logToWandB:
            wandbrun.finish()
    





