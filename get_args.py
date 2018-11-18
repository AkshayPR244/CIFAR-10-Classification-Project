# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 16:28:10 2018

@author: saket
"""

import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-is_save_model', type=int, default=0, help ='0:not save weight. 1: save weight')
    
    parser.add_argument('-img_size', type=int, default=32, help='width or height of a single image')
    parser.add_argument('-img_depth', type=int, default=3, help='depth of image, number of channels')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size of tensor during training/testing')
    
    parser.add_argument('-lr', type=float, default=0.0002, help='constant learning rate value for adam:0.0002 for rmsprop:0.00005')
    parser.add_argument('-b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('-b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    
    parser.add_argument('-num_epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('-num_train', type=int, default=1, help='number of training cycles every batch')
    parser.add_argument('-classifier', choices=['randomforest','CNN'], default='CNN', help='Type of classfier being used')
    
    opt = parser.parse_args()
    print(opt)

    return opt