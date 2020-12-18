# -*- coding: utf-8 -*-

import argparse
import sys

def getOptions(args=sys.argv[1:]):
    

    model_names = ['preact_resnet32', 'resnet34']
    dataset_names = ['CIFAR10', 'CIFAR100', 'CUB']
    optimizer_names = ['Adam','SGD']
    noise_options = ['clean','symmetric','asymmetric']
    
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    
    parser.add_argument('--arch', '-a', metavar='ARCH', default='preact_resnet32',choices=model_names, 
                        help='model architecture: ' +' | '.join(model_names) +' (default: preact_resnet32)')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.06, type=float,
                        metavar='H-P', help='initial learning rate')
    parser.add_argument('-d', '--dataset', default='CIFAR10', type=str,choices=dataset_names,
                        metavar='DATASET', help='dataset to use')
    parser.add_argument('--NoOutputClasses', default=10, type=int,
                        metavar='N', help='number of output classes')
    parser.add_argument('--optimizer', default='Adam', type=str,choices=optimizer_names,
                        help='optimizer to use')
    parser.add_argument('--epochs', default=100, type=int,help='number of total epochs to run')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--resume', default=False, action='store_true',
                        help='Resume execution from last model')
    parser.add_argument('--noise', default='clean', type=str, choices= noise_options,
                        help='Select noise for labels')
    parser.add_argument('--noise_rate', default=0.0, type=float,
                        help='Select noise rate for labels')    
    parser.add_argument('--alphamixup', default=1.0, type=float,help='alpha parameter in mixup')
    parser.add_argument('--sample', default=45000, type=int, help='sample size')

    options = parser.parse_args(args)

    
    return options
