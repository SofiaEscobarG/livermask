# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:29:12 2019

@author: sofia
"""
import numpy as np
import csv

# 3D slices function --Sofia
def thick_slices2(imagestack, thickness):
    x = imagestack.shape[1]
    y = imagestack.shape[2]
    
    padding = np.zeros((thickness-1, x, y))
    paddedstack = np.block([[[padding]], [[imagestack]], [[padding]]])
    
    nimages = paddedstack.shape[0]
    z = nimages - thickness + 1

    thickimagestacks = np.empty((thickness*z, x, y))

    for i in range(z):
        smallstack = np.array(paddedstack[i: i + thickness, :, :])
        #thickimagestacks = np.block([[[[thickimagestacks]]], [[[smallstack]]]])
        thickimagestacks[thickness*i: thickness*(i+1), :, :] = smallstack
    return thickimagestacks



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




#imagestack1 = np.ones((2, 3)) 
#imagestack2 = 2*imagestack1
#imagestack3 = 3*imagestack1
#imagestack4 = 4*imagestack1
#imagestack5 = 5*imagestack1
    
#imagestack = np.block([[[imagestack1]], [[imagestack2]], [[imagestack3]], [[imagestack4]], [[imagestack5]]])
    
numpydatabase = np.load('trainingdata256.npy')

dataidsfull = []
with open('./trainingdata.csv', 'r') as csvfile:
    myreader = csv.DictReader(csvfile, delimiter=',')
    for row in myreader:
       dataidsfull.append( int( row['dataid']))
       
train_index = np.array(dataidsfull )

axialbounds = numpydatabase['axialtumorbounds']
dataidarray = numpydatabase['dataid']
dbtrainindex= np.isin(dataidarray, train_index )
subsetidx_train  = np.all( np.vstack((axialbounds , dbtrainindex)) , axis=0 )

trainingsubset = numpydatabase[subsetidx_train]
x_train=trainingsubset['imagedata']

imagestack = x_train[0:10, 0:10, 0:10]
thickness = 3

thickimagestacks = thick_slices(imagestack, thickness)

print (thickimagestacks)
print(thickimagestacks.shape)