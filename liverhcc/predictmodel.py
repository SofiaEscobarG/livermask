import numpy as np
import csv
import sys
import os
import json
import keras
from keras.models import Model, Sequential
from keras.models import model_from_json, load_model
from keras.utils.np_utils import to_categorical
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
import nibabel as nib
from scipy import ndimage
from sklearn.model_selection import KFold
import skimage.transform
import tensorflow as tf
import matplotlib as mptlib
#mptlib.use('TkAgg')
import matplotlib.pyplot as plt


import settings
from setupmodel import GetOptimizer, GetLoss
from buildmodel import get_unet, thick_slices, unthick_slices, unthick
from mymetrics import dsc, dsc_l2, l1, dsc_l2_3D, dsc_int
from ista import ISTA
import preprocess


#############################
# apply model to NIFTI image
#############################
def PredictModel(model=settings.options.predictmodel, image=settings.options.predictimage, imageheader=None, outdir=settings.options.segmentation, seg=None):
  
  if (model != None and image != None and outdir != None ):
  
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    
    numpypredict, origheader, origseg = preprocess.reorient(image, segloc=seg)
    
    assert numpypredict.shape[0:2] == (settings._globalexpectedpixel,settings._globalexpectedpixel)
    
    resizepredict = preprocess.resize_to_nn(numpypredict)
    resizepredict = preprocess.window(resizepredict, settings.options.hu_lb, settings.options.hu_ub)
    resizepredict = preprocess.rescale(resizepredict, settings.options.hu_lb, settings.options.hu_ub)
    
    if settings.options.D3 or settings.options.D25: 
        dataid = np.ones((resizepredict.shape[0]))
        idx = np.array([1])
        resizepredict2 = thick_slices(resizepredict, settings.options.thickness, dataid, idx)
    else: 
        resizepredict2 = resizepredict
        
    if seg: 
        origseg = preprocess.resize_to_nn(origseg)
        origseg = preprocess.livermask(origseg)
        if settings.options.D25: 
            dataid_origseg = np.ones((origseg.shape[0]))
            origseg = thick_slices(origseg, 1, dataid_origseg, idx)
            origseg = origseg[...,0]
#        if not settings.options.D25 and not settings.options.D3: 
#            origseg = origseg.transpose((0,2,1)).astype(settings.FLOAT_DTYPE)
        
        origseg_img = nib.Nifti1Image(preprocess.resize_to_original(origseg), None)
        origseg_img.to_filename( outdir.replace('.nii', '-trueseg.nii') )
    
    ###
    ### set up model
    ###
    loaded_model=load_model(model, custom_objects={'dsc_l2':dsc_l2, 'l1':l1, 'dsc':dsc, 'dsc_int':dsc, 'ISTA':ISTA}, compile=False)
    #loaded_model.summary()
    
    if settings.options.D25: 
        segout_float = loaded_model.predict( resizepredict2)[...,0]
    else: 
        segout_float = loaded_model.predict( resizepredict2[...,np.newaxis] )[...,0]
    
    segout_int   = (segout_float >= settings.options.segthreshold).astype(settings.SEG_DTYPE)

    if settings.options.D3:
        segout_float = unthick_slices(segout_float, settings.options.thickness, dataid, idx)
        segout_int   = unthick_slices(segout_int, settings.options.thickness)
    elif settings.options.D25: 
        resizepredict = resizepredict.transpose((0,2,1))
    
    #segout_int   = preprocess.largest_connected_component(segout_int).astype(settings.SEG_DTYPE)
    
    segin_windowed     = preprocess.resize_to_original(resizepredict)
    segin_windowed_img = nib.Nifti1Image(segin_windowed, None, header=origheader)
    segin_windowed_img.to_filename(outdir.replace('.nii', '-imgin-windowed.nii'))
    
    segout_float_resize = preprocess.resize_to_original(segout_float)
    segout_float_img    = nib.Nifti1Image(segout_float_resize, None, header=origheader)
    segout_float_img.to_filename( outdir.replace('.nii', '-pred-float.nii') )

    segout_int_resize = preprocess.resize_to_original(segout_int)
    segout_int_img = nib.Nifti1Image(segout_int_resize, None, header=origheader)
    segout_int_img.to_filename( outdir.replace('.nii', '-pred-seg.nii') )
    
    if seg: 
        #score = dsc_l2_3D(origseg, segout_int)
        score = dsc_l2_3D(origseg.astype(settings.FLOAT_DTYPE), segout_float)
        print('dsc:\t', 1.0 - score)

    return segout_float_resize, segout_int_resize



############################
# test dropout + variation
##########################
def PredictDropout(model=settings.options.predictmodel, image=settings.options.predictimage, outdir=settings.options.segmentation, seg=None):
    
    if not (model != None and image != None and outdir != None ):
        return
    
    if model is None:
        model = settings.options.predictmodel
    if outdir is None:
        outdir = settings.options.segmentation

  
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    
    numpypredict, origheader, origseg = preprocess.reorient(image, segloc=seg)
    assert numpypredict.shape[0:2] == (settings._globalexpectedpixel,settings._globalexpectedpixel)

    resizepredict = preprocess.resize_to_nn(numpypredict)
    resizepredict = preprocess.window(resizepredict, settings.options.hu_lb, settings.options.hu_ub)
    resizepredict = preprocess.rescale(resizepredict, settings.options.hu_lb, settings.options.hu_ub)
    
    if seg: 
        origseg = preprocess.resize_to_nn(origseg)
        origseg = preprocess.livermask(origseg)

    # save unprocessed image_in
    img_in_nii = nib.Nifti1Image(image, None, header=origheader)
    img_in_nii.to_filename( outdir.replace('.nii', '-imgin.nii') )

    # save preprocessed image_in
    segin_windowed_img = nib.Nifti1Image(resizepredict, None)
    segin_windowed_img.to_filename( outdir.replace('.nii', '-imgin-windowed.nii') )
    
    # save true segmentation 
    if seg: 
        origseg_img = nib.Nifti1Image(origseg, None)
        origseg_img.to_filename( outdir.replace('.nii', '-seg.nii') )
    
    
    ###
    ### set up model
    ###
    
    loaded_model = load_model(model, custom_objects={'dsc_l2':dsc_l2, 'l1':l1, 'dsc':dsc, 'dsc_int':dsc, 'ISTA':ISTA})
    
    
    ###
    ### making baseline prediction and saving to file
    ###
    
    print('\tmaking baseline predictions...')
    
    
    segout_float = loaded_model.predict( resizepredict[...,np.newaxis] )[...,0]
    segout_int   = (segout_float >= settings.options.segthreshold).astype(settings.SEG_DTYPE)
    segout_int   = preprocess.largest_connected_component(segout_int).astype(settings.SEG_DTYPE)
    
    segout_float_img    = nib.Nifti1Image(segout_float, None)
    segout_float_img.to_filename( outdir.replace('.nii', '-pred-float.nii') )

    segout_int_img    = nib.Nifti1Image(segout_int, None)
    segout_int_img.to_filename( outdir.replace('.nii', '-pred-seg.nii') )
    
    if seg: 
        score = dsc_l2_3D(origseg, segout_int)
        print('\t\t\tdsc:\t', 1.0 - score)
        
    
    ###
    ### making predictions using different Bernoulli draws for dropout
    ###

    print('\tmaking predictions with different dropouts trials...')

    f = K.function([loaded_model.layers[0].input, K.learning_phase()],
                   [loaded_model.layers[-1].output])

    results    = np.zeros(resizepredict.shape + (settings.options.ntrials,))
    for jj in range(settings.options.ntrials):
        results[...,jj] = f([resizepredict[...,np.newaxis], 1])[0][...,0]

    print('\tcalculating statistics...')

    pred_avg = results.mean(axis=-1)
    pred_var = results.var(axis=-1)
    pred_ent = np.zeros(pred_avg.shape)
    ent_idx0 = pred_avg > 0
    ent_idx1 = pred_avg < 1
    ent_idx  = np.logical_and(ent_idx0, ent_idx1)
    pred_ent[ent_idx] = -1*np.multiply(      pred_avg[ent_idx], np.log(      pred_avg[ent_idx])) \
                        -1*np.multiply(1.0 - pred_avg[ent_idx], np.log(1.0 - pred_avg[ent_idx]))

    print('\tsaving trial statistics...')

    # save pred_avg
    pred_avg_img = nib.Nifti1Image(pred_avg, None)
    pred_avg_img.to_filename( outdir.replace('.nii', '-pred-avg.nii') )

    # save pred_var
    pred_var_img = nib.Nifti1Image(pred_var, None)
    pred_var_img.to_filename( outdir.replace('.nii', '-pred-var.nii') )
    
    # save pred_ent
    pred_ent_img = nib.Nifti1Image(pred_ent, None)
    pred_ent_img.to_filename( outdir.replace('.nii', '-pred-ent.nii') )

    print('\n')

    return segout_int, segout_float


