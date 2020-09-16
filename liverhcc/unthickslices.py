# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 11:28:02 2020

@author: sofia
"""
import numpy as np
import csv

# 3D slices function --Sofia
def thick_slices(imagestack, thickness, D3=1, D25=0):
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


def unthick_slices(imagestack, thickness):
    nthick = imagestack.shape[0]
    x      = imagestack.shape[1]
    y      = imagestack.shape[2]
    z      = nthick + thickness - 3
    
    unthickstack = np.zeros((x,y,z))
    imagestack = imagestack.transpose(2,1,0,3)
    print(imagestack.shape)
    print(unthickstack.shape)
    
    for i in range(z):
        if i == 0:
            unthickstack[:,:,i] = imagestack[:, :, 0, 1]
        if i == z: 
            unthickstack[:,:,i] = imagestack[:,:,z,thickness]
            
        elif i < thickness and i!=0: 
            thickavg = np.zeros((x,y,i))
            for j in range(i):
                thickavg[:, :, j] = imagestack[:, :, i+j-1, i-j-1]

        elif i >= nthick and i!=z:
            thickavg = np.zeros((x,y,z-i))
            for j in range(z-i-1):
                thickavg[:, :, j] = imagestack[:, :, i+j-1, thickness-j-1]
        else: 
            thickavg = np.zeros((x,y,thickness))
            for j in range(thickness):
                thickavg[:, :, j] = imagestack[:, :, i+j-1, thickness-j-1] 
   
        if i!=0 and i!=z:
            unthickstack[:,:,i] = np.average(thickavg, axis=2)
        print(unthickstack)    
    return unthickstack.transpose((2,0,1))
                

#A = np.ones((9, 2, 2))
thick=3
a = range(1,37)
A = np.reshape(a,newshape=(9,2,2))
print("A array")
print(A)
print(A.shape)
thickstack = thick_slices(A,thick)
print("thickstack")
print(thickstack)
print(thickstack.shape)
    
#a = np.average(A,axis=0)
#print(a)
    
unthickstack = unthick_slices(thickstack, thick)
print("unthickstack")
print(unthickstack)
print(unthickstack.shape)