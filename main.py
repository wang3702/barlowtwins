# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time

from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms
from imagenet import imagenet
from knn_monitor import knn_monitor
import ResNet as models

parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('data', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=2048, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=10, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument("--type",default=0,type=int,help="different type for BT")
parser.add_argument("--knn_freq",type=int,default=10, help="report current accuracy under specific iterations")
parser.add_argument("--knn_batch_size",type=int, default=128, help="default batch size for knn eval")
parser.add_argument("--knn_neighbor",type=int,default=200,help="nearest neighbor used to decide the labels")
parser.add_argument("--tensorboard",type=int,default=0,help="use tensorboard or not")
parser.add_argument("--group_norm_size", default=8, type=int, help="group norm size to normalize")

def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        sleep_time = random.random()
        time.sleep(sleep_time)#in order to avoid same directory collision#gen file before submitting jobs
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except:
                return False
        print(path + " created")
        return True
    else:
        print (path+' existed')
        return False
def main():
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if 'SLURM_JOB_ID' in os.environ:
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
        args.dist_url = f'tcp://{host_name}:58472'
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = 'tcp://localhost:58472'
        args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    args.rank += gpu
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        args.checkpoint_dir = os.path.join(args.checkpoint_dir,"Type_"+str(args.type))
        mkdir(args.checkpoint_dir)
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, "group_" + str(args.group_norm_size))
        mkdir(args.checkpoint_dir)
        stats_file = open(os.path.join(args.checkpoint_dir,'stats.txt'), 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = BarlowTwins(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu],find_unused_parameters=True)
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)

    # automatically resume from checkpoint if it exists
    if os.path.isfile(os.path.join(args.checkpoint_dir,'checkpoint.pth')):
        ckpt = torch.load(os.path.join(args.checkpoint_dir ,'checkpoint.pth'),
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0

    dataset = torchvision.datasets.ImageFolder(args.data / 'train', Transform())
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    testdir = os.path.join(args.data, 'val')
    traindir = os.path.join(args.data, 'train')
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    # val_dataset = datasets.ImageFolder(traindir,transform_test)
    val_dataset = imagenet(traindir, 0.2, transform_test)
    test_dataset = torchvision.datasets.ImageFolder(testdir, transform_test)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=per_device_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=per_device_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=False)
    writer = None
    if args.tensorboard:
        # tensorboard --logdir=Tensorboard --port=8081 --bind_all
        from tensorboardX import SummaryWriter
        log_tensor = os.path.join(args.checkpoint_dir, 'Tensorboard')
        if args.rank == 0:
            mkdir(log_tensor)
        time.sleep(5)
        check_dir = os.path.join(log_tensor, "Data")
        if args.rank == 0:
            mkdir(check_dir)
        time.sleep(5)
        check_dir = os.path.join(log_tensor, "data")
        if args.rank == 0:
            mkdir(check_dir)
        time.sleep(5)
        writer = SummaryWriter(log_tensor)
        params = vars(args)
        writer.add_text('Text', str(params))
    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for step, ((y1, y2), _) in enumerate(loader, start=epoch * len(loader)):
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model(y1, y2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if step % args.print_freq == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_biases=optimizer.param_groups[1]['lr'],
                                 loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
        print("gpu consuming before cleaning:", torch.cuda.memory_allocated() / 1024 / 1024)
        torch.cuda.empty_cache()
        print("gpu consuming after cleaning:", torch.cuda.memory_allocated() / 1024 / 1024)
        knn_test_acc = knn_monitor(model.module.encoder_q, val_loader, test_loader,
                                   global_k=min(args.knn_neighbor, len(val_loader.dataset)))
        if writer is not None and args.rank == 0:
            writer.add_scalars('Data/KNN_Acc', {'knn_acc': knn_test_acc}, epoch)
        print({'*KNN monitor Accuracy': knn_test_acc})
        if args.rank == 0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, os.path.join(args.checkpoint_dir, 'checkpoint.pth'))
    if args.rank == 0:
        # save final model
        torch.save(model.module.backbone.state_dict(),
                   os.path.join(args.checkpoint_dir , 'resnet50.pth'))


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class global_ops(nn.Module):
    def __init__(self):
        super(global_ops, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self,input):
        """

        Args:
            x: N*[B*C*H*W]
            feature map
        Returns:

        """
        x = self.avgpool(input)
        x = torch.flatten(x, 1)
        return x

class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder_q = models.__dict__['resnet50'](zero_init_residual=True)#torchvision.models.resnet50(zero_init_residual=True)
        self.type = args.type
        # projector
        sizes = [2048] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)
        self.projector = nn.Sequential(global_ops(),
                                       self.projector)

        # normalization layer for the representations z1 and z2
        if self.type==0 or self.type==1:
            self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        else:
            self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
            from ops.convert_model_togroupmodel import convert_model_to_group
            self.bn = convert_model_to_group(self.args.world_size, self.args.group_norm_size, self.bn)
            self.sync_bn = nn.BatchNorm1d(sizes[-1], affine=False)
            self.sync_bn = nn.SyncBatchNorm.convert_sync_batchnorm(self.sync_bn)
    def forward_baseline(self, y1, y2):
        z1 = self.projector(self.encoder_q(y1))
        z2 = self.projector(self.encoder_q(y2))

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss

    def forward_detach_baseline(self, y1, y2):
        z1 = self.projector(self.encoder_q(y1))
        z2 = self.projector(self.encoder_q(y2))

        # empirical cross-correlation matrix
        c1 = self.bn(z1).T @ self.bn(z2.detach())
        c2 = self.bn(z1.detach()).T @ self.bn(z2)
        c = (c1 + c2) / 2
        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss

    def forward_loco(self, y1, y2):
        z1 = self.projector(self.encoder_q(y1))
        z2 = self.projector(self.encoder_q(y2))

        # empirical cross-correlation matrix
        c1 = self.bn(z1).T @ self.sync_bn(z2.detach())
        c2 = self.sync_bn(z1.detach()).T @ self.bn(z2)
        c = (c1 + c2) / 2
        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss
    def forward(self,y1,y2):
        if self.type==0:
            return self.forward_baseline(y1,y2)
        elif self.type==1:
            return self.forward_detach_baseline(y1, y2)
        elif self.type == 2:
            return self.forward_loco(y1, y2)

class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    #print(p.shape)
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])



class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


if __name__ == '__main__':
    main()
