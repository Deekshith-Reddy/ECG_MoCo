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


    pretrain = True,
    start_epoch = 0,
    pretrain_epochs = 5,
    lr=0.03,
    momentum = 0.9,
    weight_decay = 1e-4,
    checkpoint_freq = 15,
    schedule = [30, 60],
    print_freq = 20,

    sex_classification = False,
    
    # finetuning_epochs = [40, 30, 30, 30, 30, 20],
    finetuning_epochs = [200, 150, 150, 80, 50, 50],
    finetuning_ratios = [0.005, 0.01, 0.05, 0.10, 0.5, 1.0],
    lossParams = dict(learningRate = 3*1e-6, threshold=40., type='binary cross entropy'),
    pretrained = 'checkpoints/checkpoint_sex.pth.tar',
    freeze_features = False,
    baseline = False,
        
    batch_size = 32,
    mlp = False,
    logtowandb = True,
    cos = True,

    grid_search = dict(
        aug = loader.MagnitudeWarping,
        params=dict(
            sigma = [0.02, 0.2, 2.0],
            knots = [2, 4, 8, 16]
        )
    ),

    checkpoint_dir = '/usr/sci/cibc/ProjectsAndScratch/DeekshithMLECG/checkpoints'



)

sex_args = dict (
    sex_classification=True,
    batch_size = 256,
    workers = 32,
    logtowandb=True,
    lossParams = dict(learningRate = 1e-4, threshold=40., type='binary cross entropy'),

    mlp=False,
    num_epochs=25,
    schedule=[10, 20],
    cos=True,

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
        np.random.seed(args["seed"])

        tch.cuda.manual_seed(args["seed"])
        tch.cuda.manual_seed_all(args["seed"])

        cudnn.deterministic = True
        cudnn.benchmark = False
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
    elif args["sex_classification"]:
        sex_classifcation.main_worker(sex_args)
    else:
        main_lincls.main_worker(args)

    print("DUNDANADUN")


if __name__ == "__main__":
    main()







