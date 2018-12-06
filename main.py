# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 16:23:01 2018

@author: saket
"""

import numpy as np
import os
import torch
import csv
#import torchvision.transforms as transforms
from torchvision.utils import save_image
#from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from get_args import get_args

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


if __name__ == '__main__':
    
    # give each run a random ID
    identity = str(np.random.random())[2:8]
    print('[ID]', identity)
    
    #list of hyperparameters
    args = get_args()
    
    #directory for saving loss log
    os.makedirs('logbook', exist_ok=True)

    #img_shape = (args.img_depth, args.img_size, args.img_size)
    cuda = True if torch.cuda.is_available() else False

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    device = torch.device("cuda" if cuda else "cpu")

    # Loss function, we just use BCE this time.
    CELoss = torch.nn.CrossEntropyLoss()
    #BCELoss = torch.nn.BinarycrossEntropyLoss()
    
    ##########################################################
    ## SETUP CLASSIFIER STRUCTURE ############################
    ##########################################################
    if args.classifier == 'CNN':
        from classifiermodel import CNNModel as Classifier
    elif args.classifier == 'ResNet':
        from classifiermodel import ResNetModel as Classifier
    elif args.classifier == 'SelfAtt':
        from classifiermodel import SelfAttentionModel as Classifier
        
    
    classifier = Classifier(args)
    #classifier = models.resnet18(pretrained = True)
    
    if cuda:
        classifier.cuda()
        CELoss.cuda()
        #print('models', classifier)
    else:
        print('models', classifier)
        
        
    # Initialize weights
    classifier.apply(weights_init_normal)
    
    
    ############################################################
    ######## DATALOADER ########################################
    ############################################################
    os.makedirs('./data/cifar10', exist_ok=True)
    train_dataloader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data/cifar10', train=True, download=True,
                           transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])),
            batch_size=args.batch_size, shuffle=True)

    test_dataloader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data/cifar10', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                           ])),
            batch_size=args.batch_size, shuffle=True)
        
    
    #define optimizer
    if args.optimizer == 'Adam':
        optimizer   = torch.optim.Adam(classifier.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    #initialize tensor
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    ######################################################
    ########### TRAINING #################################
    ######################################################
    with open('logbook/'+identity+'.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for epoch in range(args.num_epochs) :
            
            train_loss = 0
            correct = 0
            total = 0
            #load data once for each epoch
            for i, (imgs, labels) in enumerate(train_dataloader):
                
                imgs = imgs.to(device)
                labels = labels.to(device)
                #check for batch size disorder
                if labels.shape[0]!=args.batch_size:
                    continue
                
                #set image input and label input, set because its  a batch
                imgset = Variable(imgs.type(Tensor), requires_grad=True)
                labelset = Variable(labels.type(Tensor), requires_grad=True)
                
                #intialize optimizer
                optimizer.zero_grad()
                
                for idx in range(args.num_train):
                    #keep 1 for now, more can be checked
                    #number of times the classifier trains on each batch of image
                    output = classifier(imgs)
                    loss = CELoss(output,labels)
                    loss.backward(retain_graph=False)
                    optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                batches_done = epoch * len(train_dataloader) + i
                #saving log
                if batches_done == 0:
                    filewriter.writerow(['Batchnumber','Loss','Accuracy'])
                    filewriter.writerow([batches_done,(train_loss/(i+1)),(100.*correct/total)])
                else:
                    filewriter.writerow([batches_done,(train_loss/(i+1)),(100.*correct/total)])
                #print progress
                if i%100 == 0:
                    print ("[Epoch %d/%d] [Batch %d/%d] [TrainingLoss: %f] [Accuracy: %.3f]" % (epoch, args.num_epochs, i, len(train_dataloader),(train_loss/(i+1)),(100.*correct/total)))
                    
                
    ##########################################################
    ################# TESTING ################################
    ##########################################################
    loss_count =0
    count =0
    test_loss = 0
    correct = 0
    total = 0
    acc = 0
    for i, (imgs, labels) in enumerate(test_dataloader):
         
         imgs = imgs.to(device)
         labels = labels.to(device)
         #check for batch size disorder
         if labels.shape[0]!=args.batch_size:
             continue
            
         #set image input and label input, set because its  a batch
         imgset = Variable(imgs.type(Tensor), requires_grad=True)
         labelset = Variable(labels.type(Tensor), requires_grad=True)
         
         output = classifier(imgs)
         loss = CELoss(output,labels)
         loss_count+=loss
         count+=1
         
         test_loss += loss.item()
         _, predicted = output.max(1)
         total += labels.size(0)
         correct += predicted.eq(labels).sum().item()
         acc = 100.*correct/total
         
    print('The Loss of image classification code is,', loss_count/count)
    print('model id is', identity)
    print(args)
    
    if args.is_save_model == 1:
        print('Saving..')
        state = {
            'net': classifier.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
