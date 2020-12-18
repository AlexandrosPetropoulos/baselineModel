# -*- coding: utf-8 -*-

#clear the console and remove all variables present on the namespace at the same time
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import argparse
import sys
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F

import numpy as np
from PIL import Image
import os
import os.path
import pickle
import matplotlib.pyplot as plt


from utils.parsearguments import getOptions
from utils.cifar10v3 import Cifar10Dataset
from utils.mypreactresnet import resnet32


#define global variable options
options = None

def create_sets(dataset):

    global options

    if(dataset == 'CIFAR10'):
        trainset = Cifar10Dataset(root='./', train=0, transform=Cifar10Dataset.transform1,noise = options.noise, rate = options.noise_rate, sample = options.sample)
        testset = Cifar10Dataset(root='./', train=1,transform=Cifar10Dataset.transform2, noise = options.noise, rate = options.noise_rate, sample = options.sample)# agnoeite, apla mpike opos PENCIL
        valset = Cifar10Dataset(root='./', train=2, transform=Cifar10Dataset.transform2, noise = options.noise, rate = options.noise_rate, sample = options.sample)# agnoeite, apla mpike opos PENCIL
    elif(dataset == 'CIFAR100'):
        trainset = Cifar100Dataset(root='./', train=0, transform=Cifar100Dataset.transform1)
        testset = Cifar100Dataset(root='./', train=1,transform=Cifar100Dataset.transform2)
        valset = Cifar100Dataset(root='./', train=2, transform=Cifar100Dataset.transform2)
    else:
        print('not supported datset')
        
    return (trainset,testset,valset)

def new_model(arch, NoOutputClasses):
    
    if(arch == 'preact_resnet32'):
        model = resnet32()
    elif(arch == 'resnet34'):
        model = torch.hub.load('pytorch/vision', 'resnet34', pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NoOutputClasses)

    return model

def new_optimizer(model , optimizer, lr, mom, wd):
    
    if(optimizer == 'Adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    elif(optimizer == 'SGD'):
        optimizer = torch.optim.SGD(model.parameters(), lr,momentum=mom,weight_decay=wd)
            
    return optimizer

def adjust_learning_rate(optimizer, epoch):

    """Sets the learning rate"""
    """if epoch < 200 :
        lr = options.lr
    elif epoch < 240:
        lr = 0.2
    elif epoch < 280:
        lr = 0.2/10
    else:
        lr = 0.2/100"""

    if epoch < 40 :
        lr = options.lr
    elif epoch < 80:
        lr = options.lr/10
    elif epoch < 120:
        lr = options.lr/100
    else:
        lr = options.lr/1000
   
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
def main():
    
    global options

    options = getOptions(sys.argv[1:])

    # options.NoOutputClasses = 10
    # options.arch ='preact_resnet32'
    # options.dataset ='CIFAR10'
    # options.batch_size = 128
    # options.epochs = 200
    # options.lr = 0.01 #0.001 for Adam
    # options.momentum = 0.9
    # options.optimizer = 'SGD'
    # options.weight_decay = 0.0001
    # options.workers = 1#32
    #options.resume = True
    # options.noise = 'clean'
    # options.noise_rate = 0.0 #an einai clean agnoeitai
    options.dir = os.path.join(os.getcwd(), options.noise+str(options.noise_rate)+'checkpoints')

    print('For this run options are:')
    print(options)
    
    np.random.seed(2019)
    #define device to run
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")

    # create folder for statistics
    if os.path.exists(options.dir):
        #shutil.rmtree(options.dir)
        print(options.dir+' folder exists already')
    else:
      os.makedirs(options.dir)

    with open(os.path.join(options.dir,'commandline_args.txt'), 'a+') as f:
        print(vars(options), file=f)
        
    #create datasets and dataloaders 
    (trainset, testset , valset) = create_sets(options.dataset)
    
    train_loader = DataLoader(trainset, batch_size=options.batch_size,
                            shuffle=True,num_workers=options.workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=options.batch_size,
                            shuffle=False,num_workers=options.workers, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=options.batch_size,
                            shuffle=False, num_workers=options.workers, pin_memory=True)  

    
    #create model
    model = new_model(options.arch, options.NoOutputClasses)


    #data parallelism when >1 GPUs are available
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = new_optimizer(model, options.optimizer, options.lr, options.momentum, options.weight_decay)


    loss_fn = nn.CrossEntropyLoss()
    
    statistics_hist = train(model, optimizer, loss_fn, train_loader, val_loader,test_loader,options.epochs,device)

    
    return options.epochs , statistics_hist




def train(model, optimizer,  loss_fn, train_loader, val_loader, test_loader, epochs, device): 

    global options  

    epochs = options.epochs
    cudnn.benchmark = True
    statistics_hist = {'train_loss_hist':[],'train_acc_hist':[], 'val_loss_hist':[], 'val_acc_hist':[],'test_loss_hist':[],'test_acc_hist':[], 'lr_hist':[], 'test_std_hist':[], 'current_epoch':0}

    epoch=0
    while(epoch < epochs):
        
        
        # optionally resume from a checkpoint
        if (os.path.isfile(os.path.join(options.dir,'checkpoint.pth.tar')) and options.resume):
          print("=> loading checkpoint '{}'".format('checkpoint.pth.tar'))
          checkpoint = torch.load(os.path.join(options.dir,'checkpoint.pth.tar'))
          model.load_state_dict(checkpoint['model_state_dict'])
          optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
          statistics_hist = checkpoint['current_statistics_hist']
          epoch = statistics_hist['current_epoch']+1
          print("=> loaded checkpoint '{}' (epoch {})".format(os.path.join(options.dir ,'checkpoint.pth.tar'), checkpoint['epoch']))
          options.resume = False
        else:
          print("=> no checkpoint found at '{}'".format(os.path.join(options.dir ,'checkpoint.pth.tar')))

        if(options.optimizer == 'SGD'):
            adjust_learning_rate(optimizer, epoch)


        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch, epochs-1)) 

        training_loss = 0.0
        valid_loss = 0.0
        test_loss = 0.0
        model.train()

        num_correct = 0
        num_examples = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets , idx = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item()
            correct = torch.eq(torch.max(F.softmax(output, dim = 0), dim=1)[1],targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]


        training_loss /= len(train_loader)
        current_epoch_train_acc = (num_correct / num_examples) * 100

        #save train statistics history
        statistics_hist['train_loss_hist'].append(training_loss)
        statistics_hist['train_acc_hist'].append(current_epoch_train_acc)

        # scheduler.step(valid_loss)

        with torch.no_grad():
          model.eval()
          num_correct = 0
          num_examples = 0
          test_loss_list = []
          for batch in test_loader:
            inputs, targets = batch
            inputs = inputs.to(device, dtype=torch.float)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output,targets)
            test_loss += loss.data.item()
            #std
            test_loss_list.append(loss.data.item())
            correct = torch.eq(torch.max(F.softmax(output, dim = 0), dim=1)[1],targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]


          test_std = np.std(test_loss_list)
          statistics_hist['test_std_hist'].append(test_std)
          test_loss /= len(test_loader)
          current_epoch_test_acc = (num_correct / num_examples) * 100

          #save validation statistics history
          statistics_hist['test_loss_hist'].append(test_loss)
          statistics_hist['test_acc_hist'].append(current_epoch_test_acc)

        

        statistics_hist['current_epoch'] = epoch
        for param_group in optimizer.param_groups:
            statistics_hist['lr_hist'].append(param_group['lr'])
        #save best model
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                #'last_best_val': best_val_acc,
                'loss': training_loss,
                'current_statistics_hist': statistics_hist,
                'saved_options': options
                }, os.path.join(options.dir ,'checkpoint.pth.tar'))
       
        epoch_end = time.time()
        # print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f},accuracy = {:.2f}'.format(epoch, training_loss, valid_loss, current_epoch_val_acc))
        print('Epoch: {}, Training Loss: {:.2f}, Test Loss: {:.2f},Test accuracy = {:.2f}'.format(epoch, training_loss, test_loss, current_epoch_test_acc))
        epoch = epoch+1

    return statistics_hist


if __name__ == '__main__':


    epochs, statistics_hist = main()
    
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(epochs),statistics_hist['train_loss_hist'], label='Training Loss')
    plt.plot(range(epochs),statistics_hist['test_loss_hist'],label='test Loss')

    # find position of lowest validation loss
    #minposs = valid_loss.index(min(valid_loss))+1 
    #plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.autoscale()
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.show()
    fig.savefig(os.path.join(options.dir, options.dataset+options.optimizer+'loss_plot.png'), bbox_inches='tight')
    