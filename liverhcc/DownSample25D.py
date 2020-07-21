# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:53:29 2019

@author: sofia
"""


from keras.layers import Input, Conv2D, Conv3D, UpSampling2D, UpSampling3D, Lambda, SpatialDropout2D, SpatialDropout3D, Dense, Layer, Activation, BatchNormalization, MaxPool2D, MaxPool3D, concatenate, LocallyConnected2D
from keras.models import Model, Sequential
import tensorflow as tf

import numpy as np
import csv


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
        #thickimagestacks = np.block([[[[thickimagestacks]]], [[[smallstack]]]])
        thickimagestacks[i, :, :, :] = smallstack
    return thickimagestacks


def DownSample25D (model):
    z = model.shape[3]
    model_list = []
    for i in range(z):
        model_stack = MaxPool3D(pool_size=(2, 2, 1), strides=(2,2,1), data_format='channels_last')(thickimagestacks[:,:,:,i,:])
        model_list.append(model_stack)
        model = tf.stack(model_list, axis=3)
    return model

def UpSample25D (model):
    z = model.shape[0]        
    for i in range(z):
        model[i, :, :, :] = UpSampling3D(size=(2, 2, 1), data_format='channels_last')(model[i, :, :, :])
    return model




#numpydatabase = np.load('trainingdata_small256.npy')
#
#dataidsfull = []
#with open('./trainingdata_small.csv', 'r') as csvfile:
#    myreader = csv.DictReader(csvfile, delimiter=',')
#    for row in myreader:
#       dataidsfull.append( int( row['dataid']))
#       
#train_index = np.array(dataidsfull )
#
#axialbounds = numpydatabase['axialtumorbounds']
#dataidarray = numpydatabase['dataid']
#dbtrainindex= np.isin(dataidarray, train_index )
#subsetidx_train  = np.all( np.vstack((axialbounds , dbtrainindex)) , axis=0 )
#
#trainingsubset = numpydatabase[subsetidx_train]
#x_train=trainingsubset['imagedata']
#
#imagestack = x_train
#thickness = 3
#
#thickimagestacks = thick_slices(imagestack, thickness)
#print(thickimagestacks.shape)

thickimagestacks = Input(shape=(10, 10, 3, 4, 20))
print(thickimagestacks.shape)

z = thickimagestacks.shape[3]
model_list = []
for i in range(z):
    model_stack = MaxPool3D(pool_size=(2, 2, 1), strides=(2,2,1), data_format='channels_last')(thickimagestacks[:,:,:,i,:])
    model_list.append(model_stack)
    model = tf.stack(model_list, axis=3)
print(model.shape)


z = model.shape[3]
model_list = []
for i in range(z):
    model_stack = UpSampling3D(size=(2, 2, 1), data_format='channels_last')(model[:,:,:,:,i,:])
    model_list.append(model_stack)
    model = tf.stack(model_list, axis=3)
print(model.shape)

