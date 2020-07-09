import numpy as np
import keras
from keras.layers import Input, Conv2D, Conv3D, UpSampling2D, UpSampling3D, Lambda, SpatialDropout2D, SpatialDropout3D, Dense, Layer, Activation, BatchNormalization, MaxPool2D, MaxPool3D, concatenate, Add, LocallyConnected2D, DepthwiseConv2D
from keras.models import Model, Sequential
from keras.models import model_from_json, load_model
from keras.utils import multi_gpu_model
from keras.utils.np_utils import to_categorical
import keras.backend as K
from keras.initializers import Constant
import tensorflow as tf

import settings

# stacks 2D slices to create 3D slices -- Sofia 
def thick_slices(imagestack, thickness):
    x = imagestack.shape[1]
    y = imagestack.shape[2]
    
    padding = np.zeros((1, x, y))
    paddedstack = np.block([[[padding]], [[imagestack]], [[padding]]])
    
    nimages = paddedstack.shape[0]
    z = nimages - thickness + 1

    thickimagestacks = np.empty((z, thickness, x, y))

    for i in range(z):
        smallstack = np.array(paddedstack[i: i + thickness, :, :])
        thickimagestacks[i, :, :, :] = smallstack
    return thickimagestacks


def MaxPool25D (model):
    model = MaxPool3D(pool_size=(1, 2, 2), strides=(1,2,2), data_format='channels_last')(model)
    return model


def UpSample25D (model):
    model= UpSampling3D(size=(1, 2, 2), data_format='channels_last')(model)
    return model



def addConvBNSequential(model_in, filters=32):
    print(model_in.shape)
    if settings.options.batchnorm:
          model_in = BatchNormalization()(model_in)
          
    if settings.options.dropout > 0.0:
        if settings.options.D3:
            model_in = SpatialDropout3D(settings.options.dropout)(model_in)
        else:
          model_in = SpatialDropout2D(settings.options.dropout)(model_in)
          
    if settings.options.D3: 
        model = Conv3D(filters=filters, kernel_size=(3,3,3), padding='same', activation=settings.options.activation)(model_in)
    else:
        model = Conv2D(filters=filters, kernel_size=(3,3), padding='same', activation=settings.options.activation)(model_in)
    
    if settings.options.fanout:
        model = concatenate([model_in, model])
    else:
        model = Add()([model_in, model])
    return model


def module_down(model, filters=16, activation='prelu'):
    for i in range(settings.options.nu):
        model = addConvBNSequential(model, filters=filters)
    if settings.options.D3:
        model = MaxPool25D(model)
    else:
        model = MaxPool2D()(model)
    return model


def module_up(model, filters=16):
    if settings.options.reverse_up:
        for i in range(settings.options.nu):
            model = addConvBNSequential(model, filters=filters)
        if settings.options.D3: 
            model = UpSample25D(model)
        else:
            model = UpSampling2D()(model)
    else:
        if settings.options.D3:
            model = UpSample25D (model)
        else: 
            model = UpSampling2D()(model)
        for i in range(settings.options.nu):
            model = addConvBNSequential(model, filters=filters)
    return model


def module_mid(model, depth, filters=16):
    if settings.options.fanout and depth < settings.options.depth:
        filters = filters*2
    if depth==0:
        for i in range(settings.options.nu_bottom):
            model = addConvBNSequential(model, filters=filters)
        return model
    else:
        m_down = module_down(model, filters=filters)
        m_mid  = module_mid(m_down, depth=depth-1, filters=filters)
        m_up   = module_up(m_mid, filters=filters)
        if settings.options.skip:
            m_up = concatenate([model, m_up])
        else:
            m_up = Add()([model, m_up])
        return m_up    
    

def get_unet( _num_classes=1):
    _depth   = settings.options.depth
    _filters = settings.options.filters
    _v       = settings.options.v
    _nu      = settings.options.nu

    if settings.options.gpu > 1:
        with tf.device('/cpu:0'):
            if settings.options.D3: 
                layer_in  = Input(shape=(settings._ny, settings._nx, settings.options.thickness, 1))
            else: 
                layer_in  = Input(shape=(settings._ny, settings._nx, 1))
                
            layer_adj = Activation('linear')(layer_in)
            layer_adj = Lambda(lambda x: (x - 100.0) / 80.0)(layer_adj)
#            layer_adj = Conv2D(kernel_size=(1,1), filters=1, kernel_initializer=Constant(value=0.0125), bias_initializer=Constant(value=-1.25))(layer_adj)
            layer_mid = Activation('hard_sigmoid')(layer_adj)

            for j in range(_v):
                layer_mid = module_mid(layer_mid, depth=_depth, filters=_filters)
                for i in range(_nu):
                    layer_mid = addConvBNSequential(layer_mid,      filters=_filters)

            layer_out = Dense(_num_classes, activation='sigmoid', use_bias=True)(layer_mid)
            model = Model(inputs=layer_in, outputs=layer_out)
            return multi_gpu_model(model, gpus=settings.options.gpu)
    else:
        if settings.options.D3: 
            layer_in  = Input(shape=(settings.options.thickness, settings._ny, settings._nx, 1))
        else: 
            layer_in  = Input(shape=(settings._ny, settings._nx, 1))
            
        layer_adj = Activation('linear')(layer_in)
        layer_adj = Lambda(lambda x: (x - 100.0) / 80.0)(layer_adj)
#        layer_adj = Conv2D(kernel_size=(1,1), filters=1, kernel_initializer=Constant(value=0.0125), bias_initializer=Constant(value=-1.25))(layer_adj)
        layer_mid = Activation('hard_sigmoid')(layer_adj)

        for j in range(_v):
            layer_mid = module_mid(layer_mid, depth=_depth, filters=_filters)
            for i in range(_nu):
                layer_mid = addConvBNSequential(layer_mid,      filters=_filters)

#        if settings.options.dropout > 0.0:
#            if settings.options.D3:
#                layer_mid = SpatialDropout3D(settings.options.dropout)(layer_mid)
#            else:
#              layer_mid = SpatialDropout2D(settings.options.dropout)(layer_mid)
        
        layer_out = Dense(_num_classes, activation='sigmoid', use_bias=True)(layer_mid)
        
        model = Model(inputs=layer_in, outputs=layer_out)
        
        return model






