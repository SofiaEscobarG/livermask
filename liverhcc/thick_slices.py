# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:29:12 2019
Updated on Wed Oct 7  12:31:17 2020

@author: sofia
"""
import numpy as np

# 3D slices function --Sofia FIXED PADDING 
def thick_slicesv2(imagestack, thickness, dataid, idx, D3=False, D25=False):
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



## ORIGINAL THICKSLICES FUNCTION ================================================
#def thick_slices(imagestack, thickness):
#    x = imagestack.shape[1]
#    y = imagestack.shape[2]
#    
#    if settings.options.D3:
#        w = 1
#    elif settings.options.D25:
#        w = thickness//2
#    
#    padding = np.zeros((w, x, y))
#    paddedstack = np.vstack((padding, imagestack, padding))
#    
#    nimages = paddedstack.shape[0]
#    z = nimages - thickness + 1
#    
#    #paddedstack = np.reshape(paddedstack, (x,y,nimages))
#    paddedstack = np.transpose(paddedstack,(2,1,0))
#    thickimagestacks = np.empty((z, x, y, thickness))
#
#    for i in range(z):
#        thickimagestacks[i, :, :, :] = paddedstack[:,:,i: i + thickness]
#    return thickimagestacks
## ==============================================================================


thick=3
a = range(1,37)
dataid = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
idx = np.array([1, 2, 3])
#b = range(13,25)
A = np.reshape(a,newshape=(9,2,2))
#B = np.reshape(b,newshape=(3,2,2))
print(A)
#print(B)
#C = np.vstack((A,B))
#print(C)

A = thick_slicesv2(A, thick, dataid, idx, D3=True)
print(A)
print(A.shape)
#A = A[...,0]
#print(A.shape)