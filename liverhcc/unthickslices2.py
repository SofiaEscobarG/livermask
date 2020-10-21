# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:05:39 2020

@author: sofia
"""
import numpy as np 

# 3D slices function --Sofia
def thick_slices(imagestack, thickness, dataid, idx, D3=False, D25=False):
    x      = imagestack.shape[1]    # num x pixels 
    y      = imagestack.shape[2]    # num y pixels 
    nslice = imagestack.shape[0]    # total num of slices in imagestack
        
    if D3:
        w = 1 
    elif D25: 
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
        
    if D3: 
        thickimagestacks = thickimagestacks[0:track,:,:,:]
    return thickimagestacks


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
def unthick(thickimagestack, thickness, dataid, idx, D3=False, D25=False):
    (z,x,y,_) = thickimagestack.shape
    
    unthickstack = np.zeros((dataid.shape[0], x, y))
    
    if D3:
        npadding = 2
    elif D25:
        npadding = (thickness//2)*2
    
    track1 = 0
    track2 = 0
    
    for ii in idx:
        volumeslices = sum(np.isin(dataid, ii)) + npadding
        thickslices = volumeslices - thickness + 1
        print(volumeslices)
        
        volstack = thickimagestack[track1:(track1+thickslices),:,:,:]
        print(volstack.shape)
        
        small_stack = unthick_slices(volstack, thickness)
        track1 = track1 + thickslices
        
        unthickstack[track2:(track2+small_stack.shape[0]),:,:] = small_stack
        track2 = track2 + small_stack.shape[0]
        
        if track1 >= z:
            break
    
    return unthickstack
                
                
thick=3
a = range(1,161)
dataid = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
idx = np.array([1, 2, 3, 4])

A = np.reshape(a,newshape=(40,2,2))
print("A array")
print(A.shape)
thickstack = thick_slices(A, thick, dataid, idx, D3=True)
print("thickstack")
print(thickstack)
print(thickstack.shape)
    
#a = np.average(A,axis=0)
#print(a)
    
unthickstack = unthick(thickstack, thick, dataid, idx, D3=True)
print("unthickstack")
print(unthickstack)
print(unthickstack.shape)  