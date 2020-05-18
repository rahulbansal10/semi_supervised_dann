import argparse
import os
import shutil
import time
import importlib
import pickle

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets110
import torchvision.models as models
from PIL import Image, ImageOps
import numpy as np

import TCL as TCL
from model import CNN
from utils import *
import caffe_transform as caffe_t
from data import ImageList
import pdb

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ACAN Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-c', '--classes', default=31, type=int, metavar='N',
                    help='number of classes (default: 31)')
parser.add_argument('-bc', '--bottleneck', default=256, type=int, metavar='N',
                    help='width of bottleneck (default: 256)')
parser.add_argument('--gpu', default='0', type=str, metavar='N',
                    help='visible gpu')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--gamma', default=10.0, type=float, metavar='M',
                    help='dloss weight')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--train-iter', default=50000, type=int,
                    metavar='N', help='')
parser.add_argument('--test-iter', default=300, type=int,
                    metavar='N', help='')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--alpha', default=10.0, type=float, metavar='M')
parser.add_argument('--beta', default=0.75, type=float, metavar='M')
parser.add_argument('-hl', '--hidden', default=1024, type=int, metavar='N',
                    help='width of hiddenlayer (default: 1024)')
parser.add_argument('--name', default='alexnet', type=str)

parser.add_argument('--dataset', default='None', type=str)
parser.add_argument('--traindata', default='None', type=str)
parser.add_argument('--valdata', default='None', type=str)

parser.add_argument('--traded', default=1.0, type=float)
parser.add_argument('--tradet', default=1.0, type=float)

parser.add_argument('--total_iter',default=10000.,type=float)


def main():
    global args
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    # Data loading code
    if args.dataset == 'office':
        traindir = './office_list/'+args.traindata+'_list.txt'
        targetdir = './office_list/'+'labeled_'+args.valdata+'.txt'
        valdir = './office_list/'+'unlabeled_'+args.valdata+'.txt'

    pdb.set_trace()
    data_transforms = {
      'train': caffe_t.transform_train(resize_size=256, crop_size=224),
      'val': caffe_t.transform_train(resize_size=256, crop_size=224),
    }
    data_transforms = caffe_t.transform_test(data_transforms=data_transforms, resize_size=256, crop_size=224)
    
    source_dataset = ImageList(open(traindir).readlines(), open(targetdir).readlines(), domain = "train", transform = data_transforms["train"])
    source_loader = torch.utils.data.DataLoader(source_dataset,batch_size=args.batch_size, shuffle=True,num_workers=args.workers)

    target_dataset = ImageList(open(valdir).readlines(), domain = "val", transform = data_transforms["val"])
    target_loader = torch.utils.data.DataLoader(target_dataset,batch_size=args.batch_size, shuffle=True,num_workers=args.workers)
    
    val_dataset = ImageList(open(valdir).readlines(), domain = "val", transform = data_transforms["val9"])
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size, shuffle=False,num_workers=args.workers)

    pdb.set_trace()

    net = TCL.TCL_Net(args).cuda()
    TCL.train_val(source_loader, target_loader, val_loader,net, args)

    


if __name__ == '__main__':
    main()
