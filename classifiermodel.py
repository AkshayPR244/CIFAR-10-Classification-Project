# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 17:35:09 2018

@author: saket
"""


import torch.nn as nn
import torch.nn.functional as F
#import torch

#classifier class
class CNNModel(nn.Module):
    def __init__(self, args):
        super(CNNModel, self).__init__()
        
        #initialize arguments
        self.args      = args
        self.img_shape = (args.img_depth, args.img_size, args.img_size)
        
        def block(in_feat, out_feat, kernel_size=3, normalize=True, pooling=True):
            layers = [  nn.Conv2d(in_feat, out_feat, kernel_size, stride = 1, padding = 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            if pooling:
                layers.append(nn.MaxPool2d(2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.conv_1 = nn.Sequential(
                *block(self.args.img_depth,16),
                *block(16,32),
                *block(32,64),
                *block(64,32),
                *block(32,1)
                )
        
        self.activation = nn.Sequential(nn.Sigmoid())
        
    def forward(self,imgs):
        logits = self.conv_1(imgs)
        logits = logits.contiguous()
        
        output = self.activation(logits)
        logits = logits.view(-1)
        return output
    

if __name__ == '__main__':
    print('Classifiers initialized')