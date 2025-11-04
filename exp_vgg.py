"""
experiments on vgg network

1. import the model
2. setup benchmark code
3. setup the kernel basis algorithm
    - get other contributions code
    - write own code
4. evaluate the model performance after applying the kernel basis algorithm
5. improve the algorithm
"""

import os
import sys
import time
import torch
import torch.nn as nn
import pprint
import numpy as np
# import pretty_errors
# import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from main import AverageMeter, accuracy
from optimizedVGG import OptimizedVGG


def validate(val_loader, model, criterion, cpu=False, half=False, print_freq=10):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # if cpu == False:
        #     input = input.cuda()
        #     target = target.cuda()

        if half:
            input = input.half()

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


def validate_vgg19_cifar10(model_path, batch_size=128, workers=4):
    arch = 'vgg11'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss()

    checkpoint = torch.load(model_path)
    model = OptimizedVGG(model_type=arch)
    # model.features = torch.nn.DataParallel(model.features)
    # model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    top1_avg = validate(val_loader, model, criterion)
    return top1_avg


def do():
    arch = 'vgg11'
    batch_size = 128
    workers = 4

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss()

    checkpoint = torch.load('vgg16_faiss_kmeans_centroid_cifar10_10_10000.pth')
    # checkpoint = torch.load('vgg19.pth')

    model = OptimizedVGG(model_type=arch)
    # model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    model.load_state_dict(checkpoint)

    validate(val_loader, model, criterion)


def main():
    do()


if __name__ == '__main__':
    main()
