# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:29:12 2019

@author: sofia
"""
import numpy as np
import csv

# 3D slices function --Sofia
def thick_slices(imagestack, thickness, D3=False, D25=False):
    x = imagestack.shape[1]
    y = imagestack.shape[2]
    
    if D3:
        w = 1
    elif D25:
        w = thickness//2
    
    padding = np.zeros((w, x, y))
    paddedstack = np.vstack((padding, imagestack, padding))
    
    nimages = paddedstack.shape[0]
    z = nimages - thickness + 1
    
    #paddedstack = np.reshape(paddedstack, (x,y,nimages))
    paddedstack = np.transpose(paddedstack,(2,1,0))
    thickimagestacks = np.empty((z, x, y, thickness))

    for i in range(z):
        thickimagestacks[i, :, :, :] = paddedstack[:,:,i: i + thickness]
    return thickimagestacks


thick=1
a = range(1,37)
A = np.reshape(a,newshape=(9,2,2))
print(A)
print(A.shape)
A = thick_slices(A, thick, D25=True)
print(A)
print(A.shape)
A = A[...,0]
print(A.shape)