# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 18:00:43 2021

@author: sofia
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Conv3D, UpSampling2D, UpSampling3D, Lambda, SpatialDropout2D, SpatialDropout3D, Dense, Layer, Activation, BatchNormalization, AveragePooling2D, AveragePooling3D, MaxPooling2D, MaxPooling3D, concatenate, Add, LocallyConnected2D, DepthwiseConv2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import model_from_json, load_model

from keras.utils import multi_gpu_model
from keras.utils.np_utils import to_categorical
from keras.regularizers import l1, l2
import keras.backend as K
from keras.initializers import Constant
from keras.engine import InputLayer
import tensorflow as tf

import tensorly as tt
import matplotlib as mptlib
import matplotlib.pyplot as plt


def reorder(indices, mode):
    """
    Reorders the elements
    Taken from http://jeankossaifi.com/blog/unfolding.html
    """
    indices = list(indices)
    element = indices.pop(mode)
    return ([element] + indices[::-1])


def my_unfold(tensor, mode=0):
    """
    Returns the mode-`mode` unfolding of `tensor`
    Taken from http://jeankossaifi.com/blog/unfolding.html
    """
    return np.transpose(tensor, reorder(range(tensor.ndim), mode)).reshape((tensor.shape[mode], -1))


def print_weight(model_path):
    model = load_model(model_path)
    
    conv_norm = list()
    conv_max = list()
    conv_min = list()
    
    depconv_norm = list()
    depconv_max = list()
    depconv_min = list()
        
    for layer in model.layers:
        print(layer.name)
        if layer.get_weights() and type(layer).__name__ != 'DepthwiseConv2D': 
            conv, _ = layer.get_weights()
            conv_norm.append(tt.norm(conv))
            conv_max.append(tt.max(conv))
            conv_min.append(tt.min(conv))
            
#            plt.figure(figsize=(10,10))
#            plt.imshow(my_unfold(conv, -1))
#            plt.colorbar()
#            plt.show()
#            plt.clf()
            
        elif layer.get_weights() and type(layer).__name__ == 'DepthwiseConv2D': 
            w = layer.get_weights()                
            depconv_norm.append(tt.norm(w[0]))
            depconv_max.append(tt.max(w[0]))
            depconv_min.append(tt.min(w[0]))
            
#            plt.figure(figsize=(20,20))
#            plt.imshow(my_unfold(w[0], -1))
#            plt.colorbar()
#            plt.show()
#            plt.clf()
            
    conv_info = [conv_norm, conv_max, conv_min]
    depconv_info = [depconv_norm, depconv_max, depconv_min]
            
    return conv_info, depconv_info

modelloclist = ['mnist_model.h5', 
                'mnist_swap_model.h5', 
                'mnist_depthwisemodel.h5']


fig = plt.figure(figsize=(30,10))

norm_fig = fig.add_subplot(1, 3, 1)
max_fig = fig.add_subplot(1, 3, 2)
min_fig = fig.add_subplot(1, 3, 3)

for jj in range(len(modelloclist)):
    modelloc = modelloclist[jj]
    print(modelloc)
    conv_info, depconv_info = print_weight(modelloc)
    norm_fig.plot(conv_info[0], label=modelloc)
    max_fig.plot(conv_info[1], label=modelloc)
    min_fig.plot(conv_info[2], label=modelloc)

norm_fig.set_title('norm')
max_fig.set_title('max')
min_fig.set_title('min')
norm_fig.legend()
max_fig.legend()
min_fig.legend()
plt.show() 
    
