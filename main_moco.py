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


def dataprep(args):
    
    dataDir = '/usr/sci/cibc/Maprodxn/ClinicalECGData/LVEFCohort/pythonData/'
    normEcgs = False

    with open('patient_splits/pre_train_patients.pkl', 'rb') as file:
        pre_train_patients = pickle.load(file)

    augmentation = [
        loader.GaussianNoise(sigma=0.5, mean=10),
        loader.MagnitudeWarping(),
    ]

    augs = loader.TwoCropsTransform(transforms.Compose(augmentation))

    augmentation_name = '_'.join([str(aug.__class__.__name__) for aug in augmentation])
    print(f"Augmentation: {augmentation_name}")
    pre_train_dataset = DataTools.PreTrainECGDatasetLoader(baseDir=dataDir,augs=augs,patients=pre_train_patients.tolist(), normalize=normEcgs)

    if args["distributed"]:
        pre_train_sampler = tch.utils.data.distributed.DistributedSampler(pre_train_dataset)
    else:
        pre_train_sampler = None
    
    pre_train_loader = tch.utils.data.DataLoader(
        pre_train_dataset,
        batch_size = args["batch_size"],
        shuffle = (pre_train_sampler is None),
        num_workers = args["workers"],
        pin_memory = True,
        sampler = pre_train_sampler,
        drop_last = True
    )
    print(f"Pre Training using {len(pre_train_loader.dataset)} ECGs")

    return augmentation_name, pre_train_sampler, pre_train_loader


def main_worker(gpu, ngpus_per_node, args):
    args["gpu"] = gpu

    if args["multiprocessing_distributed"] and args["gpu"] != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass
    
    args["rank"] = gpu
    dist.init_process_group(
        backend=args["dist_backend"],
        init_method=args["dist_url"],
        world_size=args["world_size"],
        rank = args["rank"]
    )

    #Create Model
    print("Creating Model")
    encoder = Networks.ECG_SpatioTemporalNet
    model = moco_builder.MoCo(
        encoder,
        args["moco_dim"],
        args["moco_k"],
        args["moco_m"],
        args["moco_t"],
        args["mlp"],        
    )
    print(model)


    # Data Loading
    augmentation_name, pre_train_sampler, pre_train_loader = dataprep(args)
    
    # WandB
    date = datetime.datetime.now().date()

    project = f"MLECG_MoCo_LVEF_PRETRAIN"
    notes = "Pretrain"
    config = dict(
        batch_size = args["batch_size"],
        ngpus = ngpus_per_node,
        learning_rate = args["lr"],
        epochs = args["pretrain_epochs"],
        moco_dim = args["moco_dim"],
        moco_k = args["moco_k"],
        pre_train_size = len(pre_train_loader.dataset),
        augmentation = augmentation_name,
        
    )
    networkLabel = "pre_train_ECG_SpatialTemporalNet"
    if not args["multiprocessing_distributed"] or (
            args["multiprocessing_distributed"] and args["rank"] % ngpus_per_node == 0 and args["logtowandb"]
        ):
        if args["logtowandb"]:
            wandbrun = wandb.init(
                project = project,
                notes=notes,
                tags=["training", "no artifact"],
                config=config,
                entity='deekshith',
                reinit=True,
                name=f"{networkLabel}_{datetime.datetime.now()}",
        )


    if args["distributed"]:
        if args["gpu"] is not None:
            tch.cuda.set_device(args["gpu"])
            model.cuda(args["gpu"])

            args["batch_size"] = int(args["batch_size"] / ngpus_per_node)
            args["workers"] = int((args["workers"] + ngpus_per_node - 1) / ngpus_per_node)

            model = tch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args["gpu"]]
            )
        else:
            model.cuda()
            model = tch.nn.parallel.DistributedDataParallel(model)
    elif args["gpu"] is not None:
        tch.cuda.set_device(args["gpu"])
        model = model.cuda(args["gpu"])
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    
    criterion = nn.CrossEntropyLoss().cuda(args["gpu"])

    optimizer = tch.optim.SGD(
        model.parameters(),
        args["lr"],
        momentum=args["momentum"],
        weight_decay=args["weight_decay"]
    )

    cudnn.benchmark = True

    
    
    for epoch in range(args["pretrain_epochs"]):
        if args["distributed"]:
            pre_train_sampler.set_epoch(epoch)
        lr = adjust_learning_rate(optimizer, epoch, args)

        losses, top1, top5 = train(pre_train_loader, model, criterion, optimizer, epoch, args)

        if not args["multiprocessing_distributed"] or (
            args["multiprocessing_distributed"] and args["rank"] % ngpus_per_node == 0
        ):
            if (epoch + 1)% args["checkpoint_freq"] == 0:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": "Spatio Temporal Net",
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best=False,
                    filename="checkpoints/checkpoint_{:04d}.pth.tar".format(epoch+1),
                )
            wandb.log({
                'Epoch': epoch,
                'loss': losses.avg,
                'acc@1': top1.avg,
                'acc@5': top5.avg,
                'lr':lr,
            })
    if not args["multiprocessing_distributed"] or (
            args["multiprocessing_distributed"] and args["rank"] % ngpus_per_node == 0 and args["logtowandb"]
        ):
        wandbrun.finish()

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}/{}]".format(epoch,args["pretrain_epochs"]),
    )

    model.train()

    end = time.time()

    for i, (images, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args["gpu"] is not None:
            images[0] = images[0].cuda(args["gpu"], non_blocking=True)
            images[1] = images[1].cuda(args["gpu"], non_blocking=True)
        
        output, target = model(im_q = images[0], im_k = images[1])
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args["print_freq"] == 0:
            progress.display(i)
    
    return losses, top1, top5


def save_checkpoint(state, is_best, filename="checkpoints/checkpoint.pth.tar"):
    tch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args["lr"]
    if args["cos"]:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args["pretrain_epochs"]))
    else:  # stepwise lr schedule
        for milestone in args["schedule"]:
            lr *= 0.1 if epoch >= milestone else 1.0
    print(f"Learning Rate = {lr}")
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with tch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter:

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self,val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__) 

class ProgressMeter:
    def __init__(self, num_batches, meters ,prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
    
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))
