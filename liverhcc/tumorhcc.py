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
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import Constant
import nibabel as nib
from scipy import ndimage
from sklearn.model_selection import KFold
import skimage.transform
import tensorflow as tf
import matplotlib as mptlib
#mptlib.use('TkAgg')
import matplotlib.pyplot as plt


import settings

from settings import process_options, perform_setup
(options, args) = process_options()

options.dbfile="./trainingdata_small.csv"
options.rootlocation="../../unlinked_livermask/data/LiTS"
options.trainmodel=True
options.D3=True
options.numepochs=1
options.thickness=5
options.outdir="../../../Tumor_Project/AWS_outputs/test3_padding"
options.predictmodel="../../../Tumor_Project/AWS_outputs/test3_padding/001/000/liver/modelunet.h5"
options.predictimage="../../unlinked_livermask/data/LiTS/TrainingBatch2/volume-130.nii"
options.segmentation="../../test3_padding.nii"

IMG_DTYPE, SEG_DTYPE, _globalnpfile, _globalexpectedpixel, _nx, _ny = perform_setup(options)
print('database file: %s ' % settings._globalnpfile )

from setupmodel import GetDataDictionary, BuildDB
from trainmodel import TrainModel
from predictmodel import PredictModel
from kfolds import OneKfold, Kfold


if options.builddb:
    BuildDB()
if options.kfolds > 1:
    if options.idfold > -1:
        databaseinfo = GetDataDictionary(options.dbfile)
        OneKfold(i=options.idfold, datadict=databaseinfo)
    else:
        Kfold()
if options.trainmodel and options.kfolds==1 : # no kfolds, i.e. k=1
    TrainModel()
if options.predictmodel:
    segloc=options.predictimage.replace("volume", "segmentation")
    PredictModel(seg=segloc)
if ( (not options.builddb) and (not options.trainmodel) and (not options.predictmodel) and (options.kfolds == 1)):
    print("parser error")
