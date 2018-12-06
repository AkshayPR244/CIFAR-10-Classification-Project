# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 17:35:09 2018

@author: saket
"""


import torch.nn as nn
import torch.nn.functional as F
import torch

#classifier class
class CNNModel(nn.Module):
    def __init__(self, args):
        super(CNNModel, self).__init__()
        
        #initialize arguments
        self.args      = args
        self.img_shape = (args.img_depth, args.img_size, args.img_size)
        
        #define a convolution block
        def c_block(in_feat, out_feat, kernel_size=3, normalize=True, pooling=True,dropout=0.2,pad=1):
            layers = [  nn.Conv2d(in_feat, out_feat, kernel_size, stride = 1, padding = pad)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            if pooling:
                layers.append(nn.MaxPool2d(2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(dropout))
            return layers
        
        #define a linear block
        def l_block(in_feat,out_feat,normalize=True,dropout=0.2):
            layers = [nn.Linear(in_feat,out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(dropout))
            return layers
        
        self.conv = nn.Sequential(
                *c_block(self.args.img_depth,64,pooling=False), #32x32
                *c_block(64,64,pooling=True), #16x16
                *c_block(64,128,pooling=False), #16x16
                *c_block(128,128,pooling=True), #8x8
                *c_block(128,256,pooling=False), #8x8
                *c_block(256,256,pooling=True), #4x4
                *c_block(256,512,pooling=False), #4x4
                *c_block(512,512,pooling=True) #2x2
                )
        
        self.activation = nn.Sequential(nn.Sigmoid())
        
        self.linear = nn.Sequential(
                *l_block(2048,128), #128
                *l_block(128,256), #256
                *l_block(256,512), #512
                *l_block(512,1024), #1024
                *l_block(1024,10) #logits=10
                )
        
    def forward(self,imgs):
        #convolution of image
        logits = self.conv(imgs)
        # change 3d tensor to 2d tensor
        logits = logits.view(self.args.batch_size,-1)
        #apply linear transformations
        logits = self.linear(logits)
        #final sigmoid activation to change values to [0:1]
        output = self.activation(logits)
        
        return output
    
# classifier class based on resnet structure
#its not at all the same thing but its equivalent ot the residual addition structure
class ResNetModel(nn.Module):
    def __init__(self, args):
        super(ResNetModel, self).__init__()
        
        #initialize arguments
        self.args      = args
        self.img_shape = (args.img_depth, args.img_size, args.img_size)
        
        #define a convolution block
        def c_block(in_feat, out_feat, kernel_size=3, normalize=True, pooling=True,dropout=0.2,pad=1):
            layers = [  nn.Conv2d(in_feat, out_feat, kernel_size, stride = 1, padding = pad)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            if pooling:
                layers.append(nn.MaxPool2d(2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(dropout))
            return layers
        
        #define a linear block
        def l_block(in_feat,out_feat,normalize=True,dropout=0.2):
            layers = [nn.Linear(in_feat,out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(dropout))
            return layers
        
        #define a residual generator block
        def resmap_gen(in_feat,kernel_size=1,normalize=True,pooling=True):
            layers =[ nn.Conv2d(in_feat,3,kernel_size,stride=1,padding=0)]
            if normalize:
                layers.append(nn.BatchNorm2d(3, 0.8))
            if pooling:
                layers.append(nn.MaxPool2d(2))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            return layers
            
        
        self.conv_1 = nn.Sequential(
                *c_block(self.args.img_depth,64,kernel_size=1,pooling=False,pad=0),
                *c_block(64,64,pooling=False),
                *c_block(64,128,kernel_size=1,pooling=False,pad=0)) #32x32
        
        self.res = nn.Sequential(*resmap_gen(131))
        
        self.common_conv = nn.Sequential(
                *c_block(131,64,kernel_size=1,pooling=True,pad=0),
                *c_block(64,64,pooling=False),
                *c_block(64,128,kernel_size=1,pooling=False,pad=0))
        
        self.activation = nn.Sequential(nn.Sigmoid())
        
        self.linear = nn.Sequential(
                *l_block(512,256), #128
                *l_block(256,128), #256
                *l_block(128,10) #512
                 #1024
                 #logits=10
                )
        
    def forward(self,imgs):
        #1st level convolution of image
        out = self.conv_1(imgs) #32x32
        #add residuals from before convolution
        out = torch.cat([out,imgs],dim=1)
        #generate residual for 2nd stage
        res1 = self.res(out)
        #2nd level convolution
        out = self.common_conv(out) #16x16
        #add residuals from before convolution
        out = torch.cat([out,res1],dim=1)
        #generate residual for 3rd stage
        res2 = self.res(out)
        #3rd level convolution
        out = self.common_conv(out) #8x8
        #add residuals from before convolution
        out = torch.cat([out,res2],dim=1)
        #generate residual for 4th stage
        res3 = self.res(out)
        #4th level convolution
        out = self.common_conv(out) #4x4
        #add residuals from before convolution
        out = torch.cat([out,res3],dim=1)
        #5th level convolution
        out = self.common_conv(out) #2x2 depth 128
        # change 3d tensor to 2d tensor
        out = out.view(self.args.batch_size,-1)
        #apply linear transformations
        logits = self.linear(out)
        #final sigmoid activation to change values to [0:1]
        output = self.activation(logits)
        
        return output
    
# classifier class based on resnet structure
#its not at all the same thing but its equivalent ot the residual addition structure
class SelfAttentionModel(nn.Module):
    def __init__(self, args):
        super(SelfAttentionModel, self).__init__()
        
        #initialize arguments
        self.args      = args
        self.img_shape = (args.img_depth, args.img_size, args.img_size)
        
        #define a convolution block
        def c_block(in_feat, out_feat, kernel_size=3, normalize=True, pooling=True,dropout=0.2,pad=1):
            layers = [  nn.Conv2d(in_feat, out_feat, kernel_size, stride = 1, padding = pad)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            if pooling:
                layers.append(nn.MaxPool2d(2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(dropout))
            return layers
        
        #define a linear block
        def l_block(in_feat,out_feat,normalize=True,dropout=0.2):
            layers = [nn.Linear(in_feat,out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Dropout(dropout))
            return layers
        
        self.conv_1 = nn.Sequential(
                *c_block(self.args.img_depth,32,kernel_size=1,pooling=False,pad=0),
                *c_block(32,64,pooling=False),
                *c_block(64,32,pooling=False),
                *c_block(32,self.args.img_depth,kernel_size=1,pooling=False,pad=0)) #64x128x32x32
        
        self.conv_2 = nn.Sequential(
                *c_block(3,32), #16x16
                *c_block(32,64), #8x8
                *c_block(64,128), #4x4
                *c_block(128,256)) #2x2
        
        self.linear = nn.Sequential(
                *l_block(1024,128), #128
                *l_block(128,256), #256
                *l_block(256,512), #512
                *l_block(512,1024), #1024
                *l_block(1024,10) #logits=10
                )
        
        self.activation = nn.Sequential(nn.Sigmoid())
        
    def forward(self,imgs):
        #convolve
        out = self.conv_1(imgs) #64x3x32x32
        # multiply out and img to get attention map
        out = torch.matmul(imgs,out)
        #convolve and pool till 64x256x2x2
        out = self.conv_2(out)
        # change 3d tensor to 2d tensor
        logits = out.view(self.args.batch_size,-1)
        #apply linear transformations
        logits = self.linear(logits)
        #final sigmoid activation to change values to [0:1]
        output = self.activation(logits)
        
        return output
    

if __name__ == '__main__':
    print('Classifiers initialized')
