import numpy as np
import csv
import sys
import os
import json
import keras
from keras.layers import Input, Conv2D, UpSampling2D, Lambda, SpatialDropout2D, Dense, Layer, Activation, BatchNormalization, MaxPool2D, concatenate, LocallyConnected2D
from keras.models import Model, Sequential
from keras.models import model_from_json, load_model
from keras.utils import multi_gpu_model
from keras.utils.np_utils import to_categorical
import keras.backend as K
from keras.callbacks import TensorBoard, TerminateOnNaN, ModelCheckpoint
from keras.callbacks import Callback as CallbackBase
from keras.preprocessing.image import ImageDataGenerator as ImageDataGenerator2D
from keras.initializers import Constant
from optparse import OptionParser # TODO update to ArgParser (python2 --> python3)
import nibabel as nib
from scipy import ndimage
from sklearn.model_selection import KFold
import skimage.transform
import tensorflow as tf
import matplotlib as mptlib
#mptlib.use('TkAgg')
import matplotlib.pyplot as plt


import settings 
import preprocess
from generator import customImageDataGenerator as ImageDataGenerator3D


###
### Training: build NN model from anonymized data
###
def TrainModel(idfold=0):

  from setupmodel import GetSetupKfolds, GetCallbacks, GetOptimizer, GetLoss
  from buildmodel import get_unet, thick_slices, unthick_slices, unthick

  ###
  ### set up output, logging and callbacks 
  ###

  kfolds = settings.options.kfolds

  logfileoutputdir= '%s/%03d/%03d' % (settings.options.outdir, kfolds, idfold)
  os.system ('mkdir -p ' + logfileoutputdir)
  os.system ('mkdir -p ' + logfileoutputdir + '/nii')
  os.system ('mkdir -p ' + logfileoutputdir + '/liver')
  print("Output to\t", logfileoutputdir)
  
  
   ###
   ### load data
   ###

  print('loading memory map db for large dataset')
  numpydatabase = np.load(settings._globalnpfile)
  (train_index,test_index,valid_index) = GetSetupKfolds(settings.options.dbfile, kfolds, idfold)

  print('copy data subsets into memory...')
  axialbounds = numpydatabase['axialliverbounds']
  dataidarray = numpydatabase['dataid']
  
  dbtrainindex = np.isin(dataidarray, train_index )
  dbtestindex  = np.isin(dataidarray, test_index  )
  dbvalidindex = np.isin(dataidarray, valid_index ) 
  
  subsetidx_train  = np.all( np.vstack((axialbounds , dbtrainindex)) , axis=0 )
  subsetidx_test   = np.all( np.vstack((axialbounds , dbtestindex )) , axis=0 )
  subsetidx_valid  = np.all( np.vstack((axialbounds , dbvalidindex)) , axis=0 )
  
  print(np.sum(subsetidx_train) + np.sum(subsetidx_test) + np.sum(subsetidx_valid))
  print(min(np.sum(axialbounds ),np.sum(dbtrainindex )))
  
  if np.sum(subsetidx_train) + np.sum(subsetidx_test) + np.sum(subsetidx_valid) != min(np.sum(axialbounds ),np.sum(dbtrainindex )) :
      raise("data error: slice numbers dont match")

  print('copy memory map from disk to RAM...')
  trainingsubset = numpydatabase[subsetidx_train]
  validsubset    = numpydatabase[subsetidx_valid]
  testsubset     = numpydatabase[subsetidx_test]

#  np.random.seed(seed=0)
#  np.random.shuffle(trainingsubset)
  
  ntrainslices = len(trainingsubset)
  nvalidslices = len(validsubset)

  if settings.options.D3:
      x_data  = trainingsubset['imagedata']
      y_data  = trainingsubset['truthdata']
      x_valid = validsubset['imagedata']
      y_valid = validsubset['truthdata']
      
      x_train = thick_slices(x_data, settings.options.thickness, trainingsubset['dataid'], train_index)
      y_train = thick_slices(y_data, settings.options.thickness, trainingsubset['dataid'], train_index)
      
      x_valid = thick_slices(x_valid, settings.options.thickness, validsubset['dataid'], valid_index)
      y_valid = thick_slices(y_valid, settings.options.thickness, validsubset['dataid'], valid_index)
      
      np.random.seed(seed=0)
      train_shuffle = np.random.permutation(x_train.shape[0])
      valid_shuffle = np.random.permutation(x_valid.shape[0])
      x_train = x_train[train_shuffle,...]
      y_train = y_train[train_shuffle,...]
      x_valid = x_valid[valid_shuffle,...]
      y_valid = y_valid[valid_shuffle,...]
      
  elif settings.options.D25: 
      x_data  = trainingsubset['imagedata']
      y_data  = trainingsubset['truthdata']
      x_valid = validsubset['imagedata']
      y_valid = validsubset['truthdata']
      
      x_train = thick_slices(x_data, settings.options.thickness, trainingsubset['dataid'], train_index)
      x_valid = thick_slices(x_valid, settings.options.thickness, validsubset['dataid'], valid_index)
      
      y_train = thick_slices(y_data, 1, trainingsubset['dataid'], train_index)
      y_valid = thick_slices(y_valid, 1, validsubset['dataid'], valid_index)
      
      np.random.seed(seed=0)
      train_shuffle = np.random.permutation(x_train.shape[0])
      valid_shuffle = np.random.permutation(x_valid.shape[0])
      x_train = x_train[train_shuffle,...]
      y_train = y_train[train_shuffle,...]
      x_valid = x_valid[valid_shuffle,...]
      y_valid = y_valid[valid_shuffle,...]
  
  else: 
      np.random.seed(seed=0)
      np.random.shuffle(trainingsubset)
      
      x_train=trainingsubset['imagedata']
      y_train=trainingsubset['truthdata']
      x_valid=validsubset['imagedata']
      y_valid=validsubset['truthdata']
  

#  slicesplit        = int(0.9 * totnslice)
#  TRAINING_SLICES   = slice(         0, slicesplit)
#  VALIDATION_SLICES = slice(slicesplit, totnslice )


  print("\nkfolds : ", kfolds)
  print("idfold : ", idfold)
  print("slices training   : ", ntrainslices)
  print("slices validation : ", nvalidslices)
  try:
      print("slices testing    : ", len(testsubset))
  except:
      print("slices testing    : 0")


  ###
  ### data preprocessing : applying liver mask
  ###
  y_train_typed = y_train.astype(settings.SEG_DTYPE)
  y_train_liver = preprocess.livermask(y_train_typed)
  
  x_train_typed = x_train
  x_train_typed = preprocess.window(x_train_typed, settings.options.hu_lb, settings.options.hu_ub)
  x_train_typed = preprocess.rescale(x_train_typed, settings.options.hu_lb, settings.options.hu_ub)

  y_valid_typed = y_valid.astype(settings.SEG_DTYPE)
  y_valid_liver = preprocess.livermask(y_valid_typed)
  
  x_valid_typed = x_valid
  x_valid_typed = preprocess.window(x_valid_typed, settings.options.hu_lb, settings.options.hu_ub)
  x_valid_typed = preprocess.rescale(x_valid_typed, settings.options.hu_lb, settings.options.hu_ub)

#  liver_idx = y_train_typed > 0
#  y_train_liver = np.zeros_like(y_train_typed)
#  y_train_liver[liver_idx] = 1
#
#  tumor_idx = y_train_typed > 1
#  y_train_tumor = np.zeros_like(y_train_typed)
#  y_train_tumor[tumor_idx] = 1
#
#  x_masked = x_train * y_train_liver - 100.0*(1.0 - y_train_liver)
#  x_masked = x_masked.astype(settings.IMG_DTYPE)



  ###
  ### create and run model   tf.keras.losses.mean_squared_error,
  ###
  opt                 = GetOptimizer()
  callbacks, modelloc = GetCallbacks(logfileoutputdir, "liver")
  lss, met            = GetLoss()
  model               = get_unet()
  model.compile(loss       = lss,
                metrics    = met,
                optimizer  = opt)

  print("\n\n\tlivermask training...\tModel parameters: {0:,}".format(model.count_params()))

  if settings.options.D3: 
      if settings.options.augment:
          train_datagen = ImageDataGenerator3D(
              brightness_range=[0.9,1.1],
              width_shift_range=[-0.1,0.1],
              height_shift_range=[-0.1,0.1],
              horizontal_flip=True,
              vertical_flip=True,
              zoom_range=0.1,
              fill_mode='nearest',
              preprocessing_function=preprocess.post_augment
              )
          train_maskgen = ImageDataGenerator3D()
      else:
          train_datagen = ImageDataGenerator3D()
          train_maskgen = ImageDataGenerator3D()
          
      valid_datagen = ImageDataGenerator3D()
      valid_maskgen = ImageDataGenerator3D()
  else:
      if settings.options.augment:
          train_datagen = ImageDataGenerator2D(
              brightness_range=[0.9,1.1],
              width_shift_range=[-0.1,0.1],
              height_shift_range=[-0.1,0.1],
              horizontal_flip=True,
              vertical_flip=True,
              zoom_range=0.1,
              fill_mode='nearest',
              preprocessing_function=preprocess.post_augment
              )
          train_maskgen = ImageDataGenerator2D()
      else:
          train_datagen = ImageDataGenerator2D()
          train_maskgen = ImageDataGenerator2D()
          
      valid_datagen = ImageDataGenerator2D()
      valid_maskgen = ImageDataGenerator2D()
      
 
  sd = 2  # arbitrary but fixed seed for ImageDataGenerators()
  
  if settings.options.D25:
      dataflow = train_datagen.flow(x_train_typed,
                                    batch_size=settings.options.trainingbatch,
                                    seed=sd,
                                    shuffle=True)
      maskflow = train_maskgen.flow(y_train_liver,
                                    batch_size=settings.options.trainingbatch,
                                    seed=sd,
                                    shuffle=True)
      
      validdataflow = valid_datagen.flow(x_valid_typed,
                                         batch_size=settings.options.validationbatch,
                                         seed=sd,
                                         shuffle=True)
      validmaskflow = valid_maskgen.flow(y_valid_liver,
                                         batch_size=settings.options.validationbatch,
                                         seed=sd,
                                         shuffle=True)
  else: 
      dataflow = train_datagen.flow(x_train_typed[...,np.newaxis],
                                    batch_size=settings.options.trainingbatch,
                                    seed=sd,
                                    shuffle=True)
      maskflow = train_maskgen.flow(y_train_liver[...,np.newaxis],
                                    batch_size=settings.options.trainingbatch,
                                    seed=sd,
                                    shuffle=True)
      
      validdataflow = valid_datagen.flow(x_valid_typed[...,np.newaxis],
                                         batch_size=settings.options.validationbatch,
                                         seed=sd,
                                         shuffle=True)
      validmaskflow = valid_maskgen.flow(y_valid_liver[...,np.newaxis],
                                         batch_size=settings.options.validationbatch,
                                         seed=sd,
                                         shuffle=True)
   
  train_generator = zip(dataflow, maskflow)  
  valid_generator = zip(validdataflow, validmaskflow)
      
  history_liver = model.fit_generator(
                        train_generator,
                        steps_per_epoch= ntrainslices // settings.options.trainingbatch,
                        validation_steps = nvalidslices // settings.options.validationbatch,
                        epochs=settings.options.numepochs,
                        validation_data=valid_generator,
                        callbacks=callbacks,
                        shuffle=True)



  ###
  ### make predicions on validation set
  ###
  print("\n\n\tapplying models...")
  
  if settings.options.D25:
      y_pred_float = model.predict( x_valid_typed )[...,0] #[...,settings.options.thickness] )
  else: 
      y_pred_float = model.predict( x_valid_typed[...,np.newaxis] )[...,0]
      
  y_pred_seg   = (y_pred_float >= settings.options.segthreshold).astype(settings.SEG_DTYPE)    

  if settings.options.D3:
      x_valid       = unthick(x_valid, settings.options.thickness, validsubset['dataid'], valid_index)
      y_valid       = unthick(y_valid, settings.options.thickness, validsubset['dataid'], valid_index)
      
      y_valid_liver = unthick(y_valid_liver, settings.options.thickness, validsubset['dataid'], valid_index)
      y_pred_float  = unthick(y_pred_float, settings.options.thickness, validsubset['dataid'], valid_index)
      y_pred_seg    = unthick(y_pred_seg, settings.options.thickness, validsubset['dataid'], valid_index)

  print("\tsaving to file...")
  
  trueinnii     = nib.Nifti1Image(x_valid,       None)
  truesegnii    = nib.Nifti1Image(y_valid,       None)
#  windownii     = nib.Nifti1Image(x_valid_typed, None)
  truelivernii  = nib.Nifti1Image(y_valid_liver, None)
  predsegnii    = nib.Nifti1Image(y_pred_seg, None )
  predfloatnii  = nib.Nifti1Image(y_pred_float, None)
  
  trueinnii.to_filename(    logfileoutputdir+'/nii/trueimg.nii.gz')
  truesegnii.to_filename(   logfileoutputdir+'/nii/truseg.nii.gz')
#  windownii.to_filename(    logfileoutputdir+'/nii/windowedimg.nii.gz')
  truelivernii.to_filename( logfileoutputdir+'/nii/trueliver.nii.gz')
  predsegnii.to_filename(   logfileoutputdir+'/nii/predtumorseg.nii.gz')
  predfloatnii.to_filename( logfileoutputdir+'/nii/predtumorfloat.nii.gz')

  print("t\done saving.")
  return modelloc


