#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from pysmiles import read_smiles, write_smiles
import threading
import pandas as pd
from rdkit import Chem
import numpy as np
import pandas as pd
import argparse
import builtins
import loader
loader = loader.loader
import math
import os
import random
import shutil
import time
import warnings

import moco.builder
import moco.loader
print("-")
import torch
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
from rdkit import RDLogger      
RDLogger.DisableLog('rdApp.*')
import json
with open('config.json', 'r') as f:
    config = json.load(f)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]
# BATCH_SIZE = 256

# csv = pd.read_csv("../1M.csv")
model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=32,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 32)",
)

parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--run_id",
    default="",
    type=str,
    metavar="ID",
    help="run_id for wandb",
)
# parser.add_argument(
#     "-b",
#     "--batch-size",
#     default=256,
#     type=int,
#     metavar="N",
#     help="mini-batch size (default: 256), this is the total "
#     "batch size of all GPUs on the current node when "
#     "using Data Parallel or Distributed Data Parallel",
# )
# parser.add_argument(
#     "--lr",
#     "--learning-rate",
#     default=0.03,
#     type=float,
#     metavar="LR",
#     help="initial learning rate",
#     dest="lr",
# )
parser.add_argument(
    "--schedule",
    default=[120, 160],
    nargs="*",
    type=int,
    help="learning rate schedule (when to drop lr by 10x)",
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank", default=-1, type=int, help="node rank for distributed training"
)
parser.add_argument(
    "--dist-url",
    default="tcp://224.66.41.62:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="Use multi-processing distributed training to launch "
    "N processes per node, which has N GPUs. This is the "
    "fastest way to use PyTorch for either single node or "
    "multi node data parallel training",
)

# moco specific configs:
parser.add_argument(
    "--moco-dim", default=128, type=int, help="feature dimension (default: 128)"
)

parser.add_argument(
    "--moco-m",
    default=0.999,
    type=float,
    help="moco momentum of updating key encoder (default: 0.999)",
)
parser.add_argument(
    "--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)"
)

# options for moco v2
parser.add_argument("--mlp", action="store_true", help="use mlp head")
parser.add_argument(
    "--aug-plus", action="store_true", help="use moco v2 data augmentation"
)
parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")


            

data_q = mp.Queue(maxsize=10)
def main():

    args = parser.parse_args()
    args.batch_size = config["BS"]
    args.lr = config["learning_rate"]
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    # create model
    print("=> creating model '{}'".format("GC Transformer"))



  
    
    # create a function that is a for loop for 100_000 * args.eopchs times reading from the generator and add it to a mp queue, then start this mp on background
    import sys

    class BaseModel(nn.Module):
        def __init__(self, pred):
            # num_layers: int, num_heads: int, dim: int, time: int, mz: int
            num_layers, num_heads, dim, time, mz = config["num_layers"], config["num_heads"], config["dim"], config["time"], config["mz"]
            super().__init__()
            self.pos = torch.nn.Embedding(time+1, dim)
            self.proj1 = torch.nn.Linear(mz, dim)
            self.cls = torch.nn.Parameter(torch.randn(1, 1, dim))
            self.encoder = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, batch_first=True), num_layers)

            self.proj = nn.Sequential(
                nn.Linear(dim, 2*dim),
                nn.GELU(),
                nn.Linear(2*dim, 4*dim),
                nn.GELU(),
                nn.Linear(4*dim, dim),
                nn.GELU(),
            )
        
            self.pred = nn.Sequential(
                nn.Linear(dim, 2*dim),
                nn.GELU(),
                nn.Linear(2*dim, 4*dim),
                nn.GELU(),
                nn.Linear(4*dim, dim)
            ) if pred else nn.Identity()
            
            self.head = nn.Linear(dim, config["K"])
        def forward(self, x):
            b, _, _ = x.shape
            x = self.proj1(x)
            cls = self.cls.expand(b, -1, -1)
            x = torch.cat((cls, x), dim=1)
            x += self.pos(torch.arange(x.shape[1], device=x.device))
            x = self.encoder(x)
            x = x[:,0,:]
            return self.head(self.pred(self.proj(x)))
            
    BaseModel = BaseModel
    if config["moco_k"] == "max":
        config["moco_k"] = (len(os.listdir("./2048/"))//config["BS"])*config["BS"]
        # Save config back to config.json
        with open('config.json', 'w') as f:
            json.dump(config, f)
    model = moco.builder.MoCo(
        BaseModel,
        config["K"],
        config["moco_k"],
        args.moco_m,
        args.moco_t, 
        args.mlp,
    )
    print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                
                model, device_ids=[args.gpu], find_unused_parameters=True
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None: 
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # traindir = os.path.join(args.data, "train")
   

    # train_dataset = datasets.ImageFolder(
    #     traindir, moco.loader.TwoCropsTransform(transforms.Compose(augmentation))
    # )

    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # else:
    #     train_sampler = None

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=(train_sampler is None),
    #     num_workers=args.workers,
    #     pin_memory=True,
    #     sampler=train_sampler,
    #     drop_last=True,
    # )
    wandb.init(config=config, resume=args.run_id)

    for epoch in range(args.start_epoch, config["epochs"]):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(loader, model, criterion, optimizer, epoch, args)

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            if epoch % 10 == 0:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "arch": "GC Transformer",
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best=False,
                    filename="/srv/s01/leaves-shared/marshall/2D/checkpoint_{:04d}.pth.tar".format(epoch),
                )

import wandb
import pylab 
import io
from PIL import Image
import json
def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(os.listdir("./2048/"))//config["BS"],
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, images in enumerate(train_loader()):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            # images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        output, target, within_log, within_lab = model(im_q=images[0], im_k=images[1])
        loss1 = criterion(output, target)
        loss2 = (criterion(within_log, within_lab) + criterion(within_log.t(), within_lab))/2
        loss = (loss1 + loss2)/2

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))
        # with open('config.json', 'r') as f:
        #     config2 = json.load(f)
        #     lr = config2["learning_rate"]
        wandb.log({"acc1": acc1[0],
                   "loss1":loss1.item(),
                   "loss2":loss2.item(),
                   "loss":loss.item(),
                   "lr": optimizer.param_groups[0]["lr"]})
                #    "diff":(np.cos(epoch*np.pi/config["epochs"]-np.pi)+1)*0.7/7+0.5})
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if i%100==0:
        #         pylab.clf()
        #         pylab.plot(model.module.q.t().detach().cpu().numpy())
        #         buf = io.BytesIO()
        #         pylab.savefig(buf, format='png')
        #         wandb.log({"fig":wandb.Image(Image.open(buf))})
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config["print_freq"] == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    global lr
    """Decay the learning rate based on schedule"""
    # lr = args.lr
    # re-read the config file for new lr
    with open('config.json', 'r') as f:
        config2 = json.load(f)
    lr = config2["learning_rate"]
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / config["epochs"])) 
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # print(correct.shape)
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
