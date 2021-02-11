# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 13:46:57 2020

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

import settings
from ista import ISTA
from depthwise_notebook import depthwise_factorization
from setupmodel import GetOptimizer, GetLoss
from mymetrics import dsc, dsc_l2, l1, dsc_l2_3D, dsc_int


def conv_swap(filepath, model=settings.options.predictmodel):

    old_model = load_model(model, custom_objects={'dsc_l2':dsc_l2, 'l1':l1, 'dsc':dsc, 'dsc_int':dsc, 'ISTA':ISTA})
    layer_lst = [l for l in old_model.layers]
    
    with tf.name_scope('my_scope'):
        layer_in = layer_lst[0].input
        print(layer_lst[0].name)
        print(layer_lst[1].name)
        
        layer_config = Activation.get_config(layer_lst[1])
        layer_temp = Activation.from_config(layer_config)
        layer_temp.build(layer_lst[1].input_shape)
        layer_mid = layer_temp(layer_in)
   
     
        for ii in range(2, len(layer_lst)-1):
            layer_type = type(layer_lst[ii]).__name__
            layer_next = type(layer_lst[ii+1]).__name__
            print(layer_lst[ii].name)
            
            
            if layer_type=="Conv2D" and layer_type!="MaxPooling2D" and layer_lst[ii].input_shape == layer_lst[ii].output_shape:
                conv, bias = layer_lst[ii].get_weights()  
                D, W, _, err, idx = depthwise_factorization(np.array(conv))
                print(err[idx-1])
                
                D = D[...,np.newaxis]
                W = W.T
                W = W[np.newaxis, np.newaxis, ...]
                
                if layer_next == "Add":
                    layer_temp = DepthwiseConv2D(kernel_size=layer_lst[ii].kernel_size, 
                                            padding='same', 
                                            activation='linear',
                                            use_bias=False,
                                            weights=[D])(layer_mid)
                    
                    layer_temp = Conv2D(filters=settings.options.filters, 
                                   kernel_size=(1,1), 
                                   padding='same', 
                                   activation=settings.options.activation,
                                   name='conv2Dweight_'+str(ii),
                                   weights=[W,bias])(layer_temp)
                    
                    layer_mid = Add()([layer_mid, layer_temp])
                    
                else:
                    layer_mid = DepthwiseConv2D(kernel_size=layer_lst[ii].kernel_size, 
                                            padding='same', 
                                            activation='linear',
                                            use_bias=False,
                                            weights=[D])(layer_mid)
                    
                    layer_mid = Conv2D(filters=settings.options.filters, 
                                   kernel_size=(1,1), 
                                   padding='same', 
                                   activation=settings.options.activation,
                                   name='conv2dweight_'+str(ii),
                                   weights=[W,bias])(layer_mid)
                    
            elif layer_type == "Add":
                continue
            
            else:
                if layer_type == "Conv2D":
                    layer_config = Conv2D.get_config(layer_lst[ii])
                    layer_temp = Conv2D.from_config(layer_config)
                elif layer_type == "MaxPooling2D":
                    layer_config = MaxPooling2D.get_config(layer_lst[ii])
                    layer_temp = MaxPooling2D.from_config(layer_config)
                elif layer_type == "AveragePooling2D":
                    layer_config = AveragePooling2D.get_config(layer_lst[ii])
                    layer_temp = AveragePooling2D.from_config(layer_config)
                elif layer_type == "UpSampling2D": 
                    layer_config = UpSampling2D.get_config(layer_lst[ii])
                    layer_temp = UpSampling2D.from_config(layer_config)
                elif layer_type == "SpatialDropout2D":
                    layer_config = SpatialDropout2D.get_config(layer_lst[ii])
                    layer_temp = SpatialDropout2D.from_config(layer_config)
                elif layer_type == "Dense":
                    layer_config = Dense.get_config(layer_lst[ii])
                    layer_temp = Dense.from_config(layer_config)
                else: 
                    layer_config = keras.layers.Layer.get_config(layer_lst[ii])
                    layer_temp = keras.layers.Layer.from_config(layer_config)
                
                layer_weight = layer_lst[ii].get_weights()
                layer_temp.build(layer_lst[ii].input_shape)
                if layer_weight: 
                    layer_temp.set_weights(layer_weight)
                
                if layer_next == "Add":
                    layer_temp = layer_temp(layer_mid)
                    layer_mid = Add()([layer_mid, layer_temp])
                else:
                    layer_mid = layer_temp(layer_mid)

        
        # final layer
        if layer_type == "Conv2D":
            layer_config = Conv2D.get_config(layer_lst[-1])
            layer_temp = Conv2D.from_config(layer_config)
        elif layer_type == "Dense":
            layer_config = Dense.get_config(layer_lst[-1])
            layer_temp = Dense.from_config(layer_config)
        
        layer_weight = layer_lst[-1].get_weights()
        layer_temp.build(layer_lst[-1].input_shape)
        if layer_weight: 
            layer_temp.set_weights(layer_weight)
        
        layer_out = layer_temp(layer_mid)
        
        new_model = Model(inputs=layer_in, outputs=layer_out)    
        new_model.summary()
        
        tf.keras.models.save_model(new_model, filepath)
    
    return new_model
