# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:05:39 2020

@author: sofia
"""
import numpy as np 

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


def unthick_slicesv0(thickimagestack, thickness):
    (z,x,y,_) = thickimagestack.shape
    nimages = z + thickness - 3
    
    paddedstack   = np.empty((x,y,z*thickness))
    
    for i in range(z):
        paddedstack[:,:,i*thickness:(i+1)*thickness] = thickimagestack[i, :, :, :]
        
    paddedstack = np.transpose(paddedstack, (2,1,0))
    paddedstack = paddedstack[1:-1, :, :]
    
    smallstack1 = paddedstack[0:thickness-1,:,:]
    
    idx = range(2*(thickness-1), len(paddedstack)-thickness-1, thickness)
    smallstack2 = np.empty((nimages-smallstack1.shape[0],y,x))
    for i in range(len(idx)):
        smallstack2[i,:,:] = paddedstack[idx[i],:,:]

    smallstack2[-1,:,:] = paddedstack[-1,:,:]  
    unthickstack = np.vstack((smallstack1, smallstack2))
    
    return unthickstack

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
                
                
thick=1
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