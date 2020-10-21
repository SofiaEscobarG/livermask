import numpy as np
import keras
from keras.layers import Input, Conv2D, Conv3D, UpSampling2D, UpSampling3D, Lambda, SpatialDropout2D, SpatialDropout3D, Dense, Layer, Activation, BatchNormalization, AveragePooling2D, AveragePooling3D, MaxPooling2D, MaxPooling3D, concatenate, Add, LocallyConnected2D, DepthwiseConv2D
from keras.models import Model, Sequential
from keras.models import model_from_json, load_model
from keras.utils import multi_gpu_model
from keras.utils.np_utils import to_categorical
from keras.regularizers import l1, l2
import keras.backend as K
from keras.initializers import Constant
import tensorflow as tf

import settings
from DepthwiseConv3D import DepthwiseConv3D
from ista import ISTA


# 3D slices function --Sofia FIXED PADDING 
def thick_slices(imagestack, thickness, dataid, idx):
    x      = imagestack.shape[1]    # num x pixels 
    y      = imagestack.shape[2]    # num y pixels 
    nslice = imagestack.shape[0]    # total num of slices in imagestack
        
    if settings.options.D3:
        w = 1 
    elif settings.options.D25: 
        w = thickness//2
    z = nslice
        
    thickimagestacks = np.zeros((z,x,y,thickness))
    track = 0
    
    for ii in idx:
        volume_idx = np.isin(dataid, ii)
        volume     = imagestack[volume_idx,:,:]
        
        topslice       = volume[1,:,:]
        paddingtop     = np.repeat(topslice[np.newaxis,...],  w, axis=0)
        
        bottomslice    = volume[-1,:,:]
        paddingbottom  = np.repeat(bottomslice[np.newaxis,...], w, axis=0)
        
        paddedvolume   = np.vstack((paddingtop, volume, paddingbottom))
        paddedvolume   = np.transpose(paddedvolume,(2,1,0))
        

        for jj in range(paddedvolume.shape[2] - thickness + 1):
            thickimagestacks[track+jj,:,:,:] = paddedvolume[:,:,jj:jj+thickness]
        track = track + paddedvolume.shape[2] - thickness + 1
        
    if settings.options.D3: 
        thickimagestacks = thickimagestacks[0:track,:,:,:]
    return thickimagestacks


# unthick_slices function --Sofia
def unthick_slices(thickimagestack, thickness):
    (z,x,y,_) = thickimagestack.shape
    nimages = z + thickness - 3
    
    paddedstack   = np.empty((x,y,z*thickness))
    
    for i in range(z):
        paddedstack[:,:,i*thickness:(i+1)*thickness] = thickimagestack[i, :, :, :]
        
    paddedstack = np.transpose(paddedstack, (2,1,0))
    paddedstack = paddedstack[1:-1, :, :]
    
    stack1 = np.empty((thickness-2,y,x))
    for row in range(thickness-2):
        idx = range(row, (thickness*(row+1)), thickness-1)
        stack1[row,:,:] = np.average(paddedstack[idx,:,:], axis=0)

    stack2 = np.empty((nimages-2*(thickness-2),y,x))    
    for i in range(nimages-2*(thickness-2)):
        row = range(thickness-2, (nimages-2*(thickness-2))*thickness, thickness)
        idx = range(row[i], row[i]+thickness, thickness-1)
        stack2[i,:,:] = np.average(paddedstack[idx,:,:], axis=0)
    
    stack3 = np.empty((thickness-2,y,x))
    for i in range(thickness-2):
        row = range((nimages-2*(thickness-2))*thickness+(thickness-2), len(paddedstack)-thickness+1, thickness)
        idx = range(row[i], row[i]+((thickness-1)*(thickness-i-1)), thickness-1)
        stack3[i,:,:] = np.average(paddedstack[idx,:,:], axis=0)  

    unthickstack = np.vstack((stack1, stack2, stack3))
    
    return unthickstack


# Fixing padding issue with unthick slices
def unthick(thickimagestack, thickness, dataid, idx):
    (z,x,y,_) = thickimagestack.shape
    
    unthickstack = np.zeros((dataid.shape[0], x, y))
    
    if settings.options.D3:
        npadding = 2
    elif settings.options.D25:
        npadding = (thickness//2)*2
    
    track1 = 0
    track2 = 0
    
    for ii in idx:
        volumeslices = sum(np.isin(dataid, ii)) + npadding
        thickslices = volumeslices - thickness + 1
        
        volstack = thickimagestack[track1:(track1+thickslices),:,:,:]
        
        small_stack = unthick_slices(volstack, thickness)
        track1 = track1 + thickslices
        
        unthickstack[track2:(track2+small_stack.shape[0]),:,:] = small_stack
        track2 = track2 + small_stack.shape[0]
        
        if track1 >= z:
            break
    
    return unthickstack


def DepthwiseConvBlock(model_in):
    if settings.options.fanout:
        _dm = 2
    else: 
        _dm = 1
    if settings.options.batchnorm:
        model_in = BatchNormalization()(model_in)
        
    if settings.options.regularizer:
        if settings.options.D3:
            model = DepthwiseConv3D(kernel_size=(3,3,3), 
                                    padding='same', 
                                    depth_multiplier=_dm, 
                                    activation=settings.options.activation, 
                                    depthwise_regularizer=l1(settings.options.regurizer))(model_in)
        else:
            model = DepthwiseConv2D(kernel_size=(3,3),
                                    padding='same', 
                                    depth_multiplier=_dm, 
                                    activation=settings.options.activation, 
                                    depthwise_regularizer=l1(settings.options.regurizer))(model_in)
    elif settings.options.ista > 0:
        if settings.options.D3:
            model = DepthwiseConv3D(kernel_size=(3,3,3), 
                                    padding='same', 
                                    depth_multiplier=_dm, 
                                    activation=settings.options.activation, 
                                    depthwise_regularizer=ISTA(mu=settings.options.ista*settings.options.lr))(model_in)
        else:
            model = DepthwiseConv2D(kernel_size=(3,3),
                                    padding='same', 
                                    depth_multiplier=_dm, 
                                    activation=settings.options.activation, 
                                    depthwise_regularizer=ISTA(mu=settings.options.ista*settings.options.lr))(model_in)
    else:
        if settings.options.D3:
            model = DepthwiseConv3D(kernel_size=(3,3,3), 
                                    padding='same', 
                                    depth_multiplier=_dm, 
                                    activation=settings.options.activation)(model_in)
        else:
            model = DepthwiseConv2D(kernel_size=(3,3), 
                                    padding='same', 
                                    depth_multiplier=_dm, 
                                    activation=settings.options.activation)(model_in)
    if settings.options.rescon:
        model = Add()([model_in, model])
    return model


def ConvBlock(model_in, filters=32, k=3, add=True):
    if settings.options.batchnorm:
        model_in = BatchNormalization()(model_in)          
    if settings.options.D3: 
        if settings.options.l2reg:
            model = Conv3D(filters=filters, 
                           kernel_size=(k,k,k), 
                           padding='same', 
                           activation=settings.options.activation,
                           kernel_regularizer=l1(settings.options.l1reg))(model_in)
        elif settings.options.l1reg: 
            model = Conv3D(filters=filters, 
                           kernel_size=(k,k,k), 
                           padding='same', 
                           activation=settings.options.activation,
                           kernel_regularizer=l1(settings.options.l1reg))(model_in)
        elif settings.options.ista:
            model = Conv3D(filters=filters, 
                           kernel_size=(k,k,k), 
                           padding='same', 
                           activation=settings.options.activation,
                           kernel_regularizer=ISTA(mu=settings.options.ista*settings.options.lr))(model_in)
        else:
            model = Conv3D(filters=filters, 
                           kernel_size=(k,k,k), 
                           padding='same', 
                           activation=settings.options.activation)(model_in)
    else:
        if settings.options.l2reg:
            model = Conv2D(filters=filters, 
                           kernel_size=(k,k), 
                           padding='same', 
                           activation=settings.options.activation,
                           kernel_regularizer=l1(settings.options.l1reg))(model_in)
        elif settings.options.l1reg:
            model = Conv2D(filters=filters, 
                           kernel_size=(k,k), 
                           padding='same', 
                           activation=settings.options.activation,
                           kernel_regularizer=l1(settings.options.l1reg))(model_in)
        elif settings.options.ista:
            model = Conv2D(filters=filters, 
                           kernel_size=(k,k), 
                           padding='same', 
                           activation=settings.options.activation,
                           kernel_regularizer=ISTA(mu=settings.options.ista*settings.options.lr))(model_in)
        else:
            model = Conv2D(filters=filters, 
                           kernel_size=(k,k), 
                           padding='same', 
                           activation=settings.options.activation)(model_in)   
    if add: 
        model = Add()([model_in, model])
    return model

    
def module_down(model, filters=16):
    model=ConvBlock(model,filters=filters)
    if settings.options.D3:
        model = AveragePooling3D(pool_size=(2,2,1), strides=(2,2,1), data_format='channels_last')(model)
    else:
        model = AveragePooling2D()(model)
    model = ConvBlock(model, filters=filters)
    #model = ConvBlock(model, filters=filters)
    return model


def module_up(model, filters=16):
    model = ConvBlock(model, filters=filters)
    if settings.options.D3:
        model = UpSampling3D(size=(2,2,1), data_format='channels_last')(model)
    else: 
        model = UpSampling2D()(model)
    model = ConvBlock(model, filters=filters)
    #model = ConvBlock(model, filters=filters)
    return model
   
    
def module_mid(model, depth, filters=16):
    if depth == 0:
        for i in range(settings.options.nu_bottom):
            model = ConvBlock(model, filters=filters)
        return model
    else:
        m_down = module_down(model, filters=filters)
        m_mid  = module_mid(m_down, depth=depth-1, filters=filters)
        m_up   = module_up(m_mid, filters=filters)
        
        m_up = ConvBlock(m_up, filters=filters, add=False)
        m_up = Add()([model, m_up])
        m_up = ConvBlock(m_up, filters=filters, add=False)
        return m_up
    

def get_unet( _num_classes=1):
    _depth   = settings.options.depth
    _filters = settings.options.filters
    _v       = settings.options.v
    
    if settings.options.D3: 
        shape_in = (settings._ny, settings._nx, settings.options.thickness, 1)
    elif settings.options.D25:
        shape_in = (settings._ny, settings._nx, settings.options.thickness)
    else: 
        shape_in = (settings._ny, settings._nx, 1)
    
    layer_in = Input(shape=shape_in)
    layer_mid = Activation('linear')(layer_in)
    layer_mid = ConvBlock(layer_mid, filters=_filters, add=False)

    for j in range(_v):
        layer_mid = module_mid(layer_mid, depth=_depth, filters=_filters)

    if settings.options.D3:
        if settings.options.dropout > 0.0:
            layer_mid = SpatialDropout3D(settings.options.dropout)(layer_mid)
        layer_out = Conv3D(filters=_num_classes, 
                           kernel_size=(1,1,1), 
                           padding='same', 
                           activation='sigmoid',
                           use_bias=True)(layer_mid)
    else:
        if settings.options.dropout > 0.0:
            layer_mid = SpatialDropout2D(settings.options.dropout)(layer_mid)
        layer_out = Conv2D(filters=_num_classes, 
                           kernel_size=(1,1), 
                           padding='same', 
                           activation='sigmoid',
                           use_bias=True)(layer_mid)   
        
    #layer_out = Dense(_num_classes, activation='sigmoid', use_bias=True)(layer_mid)
    
    model = Model(inputs=layer_in, outputs=layer_out)
    model.summary()
        
    if settings.options.gpu > 1:
        return multi_gpu_model(model, gpus=settings.options.gpu)
    else:
        return model
