import numpy as np
import csv
import os
import json
import keras
from keras.layers import Input, Conv2D, LocallyConnected2D, Lambda, Add, Maximum, Minimum, Multiply, Dense, Layer, Activation, BatchNormalization
from keras.models import Model, Sequential
from keras.models import model_from_json, load_model
from keras.utils.np_utils import to_categorical
import keras.backend as K
from keras.callbacks import TensorBoard, TerminateOnNaN, ModelCheckpoint
from keras.callbacks import Callback as CallbackBase
from optparse import OptionParser
import nibabel as nib
from scipy import ndimage
from sklearn.model_selection import KFold
import skimage.transform
import tensorflow as tf
import horovod.keras as hvd


# setup command line parser to control execution
parser = OptionParser()
parser.add_option( "--hvd",
                  action="store_true", dest="with_hvd", default=False,
                  help="use horovod for parallelism")
parser.add_option( "--builddb",
                  action="store_true", dest="builddb", default=False,
                  help="load all training data into npy", metavar="FILE")
parser.add_option( "--trainmodel",
                  action="store_true", dest="trainmodel", default=False,
                  help="train model on all data", metavar="FILE")
parser.add_option( "--predictmodel",
                  action="store", dest="predictmodel", default=None,
                  help="model weights (.h5) for prediction", metavar="Path")
parser.add_option( "--predictimage",
                  action="store", dest="predictimage", default=None,
                  help="image to segment", metavar="Path")
parser.add_option( "--segmentation",
                  action="store", dest="segmentation", default=None,
                  help="location for seg prediction output ", metavar="Path")
parser.add_option( "--trainingsolver",
                  action="store", dest="trainingsolver", default='adam',
                  help="setup info", metavar="string")
parser.add_option( "--dbfile",
                  action="store", dest="dbfile", default="./trainingdata.csv",
                  help="training data file", metavar="string")
parser.add_option( "--trainingresample",
                  type="int", dest="trainingresample", default=256,
                  help="resample so that model prediction occurs at this resolution", metavar="int")
parser.add_option( "--trainingbatch",
                  type="int", dest="trainingbatch", default=4,
                  help="batch size", metavar="int")
parser.add_option( "--kfolds",
                  type="int", dest="kfolds", default=1,
                  help="perform kfold prediction with k folds", metavar="int")
parser.add_option( "--idfold",
                  type="int", dest="idfold", default=-1,
                  help="individual fold for k folds", metavar="int")
parser.add_option( "--rootlocation",
                  action="store", dest="rootlocation", default='/rsrch1/ip/jacctor/LiTS/LiTS',
                  help="root location for images for training", metavar="string")
parser.add_option("--numepochs",
                  type="int", dest="numepochs", default=10,
                  help="number of epochs for training", metavar="int")
parser.add_option("--outdir",
                  action="store", dest="outdir", default='./',
                  help="directory for output", metavar="string")
parser.add_option("--nt",
                  type="int", dest="nt", default=10,
                  help="number of timesteps", metavar="int")
parser.add_option( "--randinit",
                  action="store_true", dest="randinit", default=False,
                  help="initialize u0 as random uniform. Default is constant 1", metavar="FILE")
parser.add_option( "--circleinit",
                  action="store_true", dest="circleinit", default=False,
                  help="initialize u0 as circle in lower right quadrant. Default is constant 1", metavar="FILE")
(options, args) = parser.parse_args()

# raw dicom data is usually short int (2bytes) datatype
# labels are usually uchar (1byte)
IMG_DTYPE = np.int16
SEG_DTYPE = np.uint8

if options.with_hvd:
    hvd.init()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=config))


_globalnpfile = options.dbfile.replace('.csv','%d.npy' % options.trainingresample )
_globalexpectedpixel=512
_nt = options.nt
_nx = options.trainingresample
_ny = options.trainingresample
_num_classes = 2 
print('database file: %s ' % _globalnpfile )


# build data base from CSV file
def GetDataDictionary():
  CSVDictionary = {}
  with open(options.dbfile, 'r') as csvfile:
    myreader = csv.DictReader(csvfile, delimiter=',')
    for row in myreader:
       CSVDictionary[int( row['dataid'])]  =  {'image':row['image'], 'label':row['label']}
  return CSVDictionary


# setup kfolds
def GetSetupKfolds(numfolds,idfold):
  # get id from setupfiles
  dataidsfull = []
  with open(options.dbfile, 'r') as csvfile:
    myreader = csv.DictReader(csvfile, delimiter=',')
    for row in myreader:
       dataidsfull.append( int( row['dataid']))
  if (numfolds < idfold or numfolds < 1):
     raise("data input error")
  # split in folds
  if (numfolds > 1):
     kf = KFold(n_splits=numfolds)
     allkfolds   = [ (train_index, test_index) for train_index, test_index in kf.split(dataidsfull )]
     train_index = allkfolds[idfold][0]
     test_index  = allkfolds[idfold][1]
  else:
     train_index = np.array(dataidsfull )
     test_index  = None
  print("kfold: \t",numfolds)
  print("idfold: \t", idfold)
  print("train_index:\t", train_index)
  print("test_index:\t",  test_index)
  return (train_index,test_index)

 
  # create upwind FD kernels
kXP = K.constant(np.asarray([[-1,1,0]])[:,:,np.newaxis,np.newaxis])
kXN = K.constant(np.asarray([[0,-1,1]])[:,:,np.newaxis,np.newaxis])
kYP = K.constant(np.asarray([[0],[1],[-1]])[:,:,np.newaxis,np.newaxis])
kYN = K.constant(np.asarray([[1],[-1],[0]])[:,:,np.newaxis,np.newaxis])
kXC = K.constant(np.asarray([[-1,0,1]])[:,:,np.newaxis,np.newaxis])
kYC = K.constant(np.asarray([[1],[0],[-1]])[:,:,np.newaxis,np.newaxis])
kXX = K.constant(np.asarray([[-1,2,1]])[:,:,np.newaxis,np.newaxis])
kYY = K.constant(np.asarray([[1],[2],[-1]])[:,:,np.newaxis,np.newaxis])
kXY = K.constant(np.asarray([[-1,0,1],[0,0,0],[1,0,-1]])[:,:,np.newaxis,np.newaxis])
blur = K.constant(np.asarray([[0.0625, 0.1250, 0.0625],[0.1250, 0.5000, 0.1250],[0.0625, 0.1250, 0.0625]])[:,:,np.newaxis,np.newaxis])


class ForcingFunction(Layer):

   def __init__(self, in_img, in_dims, **kwargs):
       self.image = in_img
       self.eps = 0.001
       self.dt = 1.0
       self.dims = in_dims
       self.dx = in_dims[:,0,0,:]
       self.dy = in_dims[:,1,0,:]
       self.dz = in_dims[:,2,0,:]
       self.shp = K.shape(self.image)
       self.rhp = (self.shp[0], self.shp[1]*self.shp[2], self.shp[3])
       super(ForcingFunction, self).__init__(**kwargs)

   def build(self, input_shape):
       self.edge_kernel_1 = self.add_weight(name='edge_kernel_1',
			shape=(3,3,1,16),
			initializer='normal',
			trainable=True)
#       self.edge_kernel_2 = self.add_weight(name='edge_kernel_2',
#			shape=(3,3,16,32),
#			initializer='normal',
#			trainable=True)
       self.kappa_kernel = self.add_weight(name='kappa_kernel',
			shape=(3,3,1,16),
			initializer='normal',
			trainable=True)
       self.div_kernel = self.add_weight(name='div_kernel',
			shape=(3,3,16,32),
			initializer='normal',
			trainable=True)
       self.const_kernel = self.add_weight(name='c_kernel',
                        shape=(1,1,1,1),
                        initializer='ones',
                        trainable=True)
       self.alpha = self.add_weight(name='alpha',
                        shape=(1,1,1,1),
                        initializer='ones',
                        trainable=True)
       self.beta = self.add_weight(name='beta',
                        shape=(1,1,1,1),
                        initializer='ones',
                        trainable=True)
       self.gamma = self.add_weight(name='gamma',
                        shape=(1,1,1,1),
                        initializer='ones',
                        trainable=True)
       super(ForcingFunction, self).build(input_shape)
   def call(self, u):

        ### edge detection (learned filter)
        ### g_I(x) = 1 / (1 + norm(grad(I))^2 )
#        edges = K.relu(K.conv2d(self.image, self.edge_kernel_1, padding='same'))
#        edges = K.relu(K.conv2d(edges,      self.edge_kernel_2, padding='same'))
        edges = K.conv2d(self.image, self.edge_kernel_1, padding='same')
        edges = K.square(edges)
        edges = K.sum(edges, axis=-1, keepdims=True)
        edges = K.constant(1.0) / ( edges + K.constant(1.0))

          
        ### grad( edge_detection )
        ### note : need to reshape before applying rescaling for each slice due to tf backend    
        gex = K.conv2d(edges, kXC, padding='same')   
        gex = K.reshape(gex, self.rhp)
        gex = K.batch_dot(gex , self.dx, axes=[2,1])
        grad_edges_x = K.reshape(gex, self.shp)

        gey = K.conv2d(edges, kYC, padding='same')
        gey = K.reshape(gey, self.rhp)
        gey = K.batch_dot(gey , self.dy, axes=[2,1])
        grad_edges_y = K.reshape(gey, self.shp)
        

	### transport - upwind approx to grad( edge_detection)^T grad( u )
        xp = K.conv2d(u, kXP, padding='same')
        xn = K.conv2d(u, kXN, padding='same')
        yp = K.conv2d(u, kYP, padding='same')
        yn = K.conv2d(u, kYN, padding='same')
        fxp =         K.relu(       grad_edges_x)
        fxn =  -1.0 * K.relu(-1.0 * grad_edges_x)
        fyp =         K.relu(       grad_edges_y)
        fyn =  -1.0 * K.relu(-1.0 * grad_edges_y)
        xpp = fxp*xp
        xnn = fxn*xn
        ypp = fyp*yp
        ynn = fyn*yn
        xterms = xpp + xnn
        xterms = K.reshape(xterms, self.rhp)
        xterms = K.batch_dot( xterms, self.dx, axes=[2,1])
        xterms = K.reshape(xterms, self.shp)
        yterms = ypp + ynn
        yterms = K.reshape(yterms, self.rhp)
        yterms = K.batch_dot( yterms, self.dy, axes=[2,1])
        yterms = K.reshape(yterms, self.shp)
        transport = xterms + yterms


        ### curvature kappa( u ) 
        gradu = K.conv2d(u, self.kappa_kernel, padding='same')
        norm2 = K.sum(K.square(gradu), axis=-1, keepdims=True)
        normu = K.sqrt(norm2 + K.epsilon())
        kappa = K.l2_normalize(gradu, axis=-1)
        kappa = K.conv2d(kappa, self.div_kernel, padding='same')
        kappa = K.sum(kappa, axis=-1, keepdims=True)
        kappa = kappa + K.relu(self.gamma)
        curvature = edges*kappa*normu

        return u + K.constant(self.dt)*(\
                                           K.conv2d(curvature, self.alpha, padding='same') \
                                        +  K.conv2d(transport, self.beta,  padding='same') )
   def compute_output_shape(self, input_shape):
       return input_shape

class ImageForcingFunction(Layer):

   def __init__(self, in_img, in_dims, **kwargs):
       self.dims = in_dims
       self.dx = in_dims[:,0,0,:]
       self.dy = in_dims[:,1,0,:]
       self.dz = in_dims[:,2,0,:]
       self.shp = K.shape(in_img)
       self.rhp = (self.shp[0], self.shp[1]*self.shp[2], self.shp[3])
       super(ImageForcingFunction, self).__init__(**kwargs)

   def build(self, input_shape):
       self.edge_kernel_1 = self.add_weight(name='edge_kernel_1',
			shape=(3,3,1,24),
			initializer='normal',
			trainable=True)
       self.edge_kernel_2 = self.add_weight(name='edge_kernel_2',
			shape=(3,3,24,48),
			initializer='normal',
			trainable=True)
       self.edge_kernel_3 = self.add_weight(name='edge_kernel_3',
			shape=(3,3,48,48),
			initializer='normal',
			trainable=True)
       super(ImageForcingFunction, self).build(input_shape)

   def call(self, img):

        ### edge detection (learned filter)
        ### g_I(x) = 1 / (1 + norm(grad(I))^2 )
        edges = K.relu(K.conv2d(img,   self.edge_kernel_1, padding='same'))
        edges = K.relu(K.conv2d(edges, self.edge_kernel_2, padding='same'))
        edges =        K.conv2d(edges, self.edge_kernel_3, padding='same')
        edges = K.square(edges)
        edges = K.sum(edges, axis=-1, keepdims=True)
        return K.constant(1.0) / ( edges + K.constant(1.0))

   def compute_output_shape(self, input_shape):
       return input_shape

class TimeStep(Layer):

   def __init__(self, in_edges, in_dims, **kwargs):
       self.dt = 1.0
       self.dims = in_dims
       self.dx = in_dims[:,0,0,:]
       self.dy = in_dims[:,1,0,:]
       self.dz = in_dims[:,2,0,:]
       self.shp = K.shape(in_edges)
       self.rhp = (self.shp[0], self.shp[1]*self.shp[2], self.shp[3])
       super(TimeStep, self).__init__(**kwargs)

   def build(self, input_shape):
       self.kappa_kernel = self.add_weight(name='kappa_kernel',
			shape=(3,3,1,16),
			initializer='normal',
			trainable=False)
       self.div_kernel = self.add_weight(name='div_kernel',
			shape=(3,3,16,48),
			initializer='normal',
			trainable=False)
       self.const_kernel = self.add_weight(name='c_kernel',
                        shape=(1,1,1,1),
                        initializer='ones',
                        trainable=True)
       self.alpha = self.add_weight(name='alpha',
                        shape=(1,1,1,1),
                        initializer='ones',
                        trainable=True)
       self.beta = self.add_weight(name='beta',
                        shape=(1,1,1,1),
                        initializer='ones',
                        trainable=True)
       super(TimeStep, self).build(input_shape)

   def call(self, in_array):
       
        u     = in_array[0]
        edges = in_array[1] 

        ### grad( edge_detection )
        ### note : need to reshape before applying rescaling for each slice due to tf backend    
        gex = K.conv2d(edges, kXC, padding='same')   
        gex = K.reshape(gex, self.rhp)
        gex = K.batch_dot(gex , self.dx, axes=[2,1])
        grad_edges_x = K.reshape(gex, self.shp)

        gey = K.conv2d(edges, kYC, padding='same')
        gey = K.reshape(gey, self.rhp)
        gey = K.batch_dot(gey , self.dy, axes=[2,1])
        grad_edges_y = K.reshape(gey, self.shp)
        

	### transport - upwind approx to grad( edge_detection)^T grad( u )
        xp = K.conv2d(u, kXP, padding='same')
        xn = K.conv2d(u, kXN, padding='same')
        yp = K.conv2d(u, kYP, padding='same')
        yn = K.conv2d(u, kYN, padding='same')
        fxp =         K.relu(       grad_edges_x)
        fxn =  -1.0 * K.relu(-1.0 * grad_edges_x)
        fyp =         K.relu(       grad_edges_y)
        fyn =  -1.0 * K.relu(-1.0 * grad_edges_y)
        xpp = fxp*xp
        xnn = fxn*xn
        ypp = fyp*yp
        ynn = fyn*yn
        xterms = xpp + xnn
        xterms = K.reshape(xterms, self.rhp)
        xterms = K.batch_dot( xterms, self.dx, axes=[2,1])
        xterms = K.reshape(xterms, self.shp)
        yterms = ypp + ynn
        yterms = K.reshape(yterms, self.rhp)
        yterms = K.batch_dot( yterms, self.dy, axes=[2,1])
        yterms = K.reshape(yterms, self.shp)
        transport = xterms + yterms


        ### curvature kappa( u ) 
        gradu = K.conv2d(u, self.kappa_kernel, padding='same')
        norm2 = K.sum(K.square(gradu), axis=-1, keepdims=True)
        normu = K.sqrt(norm2 + K.epsilon())
        kappa = K.l2_normalize(gradu, axis=-1)
        kappa = K.conv2d(kappa, self.div_kernel, padding='same')
        kappa = K.sum(kappa, axis=-1, keepdims=True)
        kappa = kappa + K.relu(self.const_kernel)
        curvature = edges*kappa*normu

        self.alpha = K.sqrt(K.square(self.alpha) + K.epsilon())
        self.beta  = K.sqrt(K.square(self.beta)  + K.epsilon())

        return u + K.constant(self.dt)*(\
                                           K.conv2d(curvature, self.alpha, padding='same') \
                                        +  K.conv2d(transport, self.beta,  padding='same') )
   def compute_output_shape(self, input_shape):
       return input_shape


def get_upwind_transport_net(_nt, _final_sigma='sigmoid'):

    in_imgs   = Input(shape=(_ny,_nx,2))
    in_img    = Lambda(lambda x : x[...,0][...,None])(in_imgs)   # I
    in_layer  = Lambda(lambda x : x[...,1][...,None])(in_imgs)   # u0
    in_dims   = Input(shape=(3,1,1))
    mid_layer = Conv2D(1, (3,3), padding='same', use_bias=True)(in_layer) 

    # Forcing Function F depends on image and on  u, but not on time
    F = ForcingFunction(in_img, in_dims)
    for ttt in range(_nt):
        mid_layer = F(mid_layer)
        
#    g_I      = ImageForcingFunction(in_img, in_dims)(in_img)
#    timestep = TimeStep(g_I, in_dims)
#    for ttt in range(_nt):
#        mid_layer = timestep([mid_layer, g_I]) 

    out_layer = Conv2D(1, (5,5), padding='same', use_bias=True)(mid_layer)
    out_layer = Activation(_final_sigma)(out_layer)
    model = Model([in_imgs, in_dims], out_layer)
    return model




# dsc = 1 - dsc_as_l2
def dsc_as_l2(y_true, y_pred, smooth=0.00001):
    numerator = K.sum(K.square(y_true[...] - y_pred[...]),axis=(1,2)) + smooth
    denominator = K.sum(K.square(y_true[...]),axis=(1,2)) + K.sum(K.square(y_pred[...]),axis=(1,2)) + smooth
    disc = numerator/denominator
    return disc # average of dsc0,dsc1 over batch/stack 
def dice_metric_zero(y_true, y_pred):
    batchdiceloss =  dsc_as_l2(y_true, y_pred)
    return 1.0 - batchdiceloss[:,0]




##########################
# preprocess database and store to disk
##########################
def BuildDB():
  # create  custom data frame database type
  mydatabasetype = [('dataid', int),
     ('axialliverbounds',bool),
     ('axialtumorbounds',bool),
     ('imagepath','S128'),
     ('imagedata','(%d,%d)int16' %(options.trainingresample,options.trainingresample)),
     ('truthpath','S128'),
     ('truthdata','(%d,%d)uint8' % (options.trainingresample,options.trainingresample)),
     ('image_dx', float),
     ('image_dy', float),
     ('image_dz', float)     ]

  # initialize empty dataframe
  numpydatabase = np.empty(0, dtype=mydatabasetype  )

  # load all data from csv
  totalnslice = 0
  with open(options.dbfile, 'r') as csvfile:
    myreader = csv.DictReader(csvfile, delimiter=',')
    for row in myreader:
      imagelocation = '%s/%s' % (options.rootlocation,row['image'])
      truthlocation = '%s/%s' % (options.rootlocation,row['label'])
      print(imagelocation,truthlocation )

      # load nifti file
      imagedata = nib.load(imagelocation )
      numpyimage= imagedata.get_data().astype(IMG_DTYPE )
      # error check
      assert numpyimage.shape[0:2] == (_globalexpectedpixel,_globalexpectedpixel)
      nslice = numpyimage.shape[2]
      resimage=skimage.transform.resize(numpyimage,
            (options.trainingresample,options.trainingresample,nslice),
            order=0,
            mode='constant',
            preserve_range=True).astype(IMG_DTYPE)

      # load nifti file
      truthdata = nib.load(truthlocation )
      numpytruth= truthdata.get_data().astype(SEG_DTYPE)
      # error check
      assert numpytruth.shape[0:2] == (_globalexpectedpixel,_globalexpectedpixel)
      assert nslice  == numpytruth.shape[2]
      restruth=skimage.transform.resize(numpytruth,
              (options.trainingresample,options.trainingresample,nslice),
              order=0,
              mode='constant',
              preserve_range=True).astype(SEG_DTYPE)

      # bounding box for each label
      if( np.max(restruth) ==1 ) :
        (liverboundingbox,)  = ndimage.find_objects(restruth)
        tumorboundingbox  = None
      else:
        (liverboundingbox,tumorboundingbox) = ndimage.find_objects(restruth)

      if( nslice  == restruth.shape[2]):
        # custom data type to subset
        datamatrix = np.zeros(nslice  , dtype=mydatabasetype )

        # custom data type to subset
        datamatrix ['dataid']                         = np.repeat(row['dataid'], nslice )
        # id the slices within the bounding box
        axialliverbounds                              = np.repeat(False, nslice )
        axialtumorbounds                              = np.repeat(False, nslice )
        axialliverbounds[liverboundingbox[2]]         = True
        if (tumorboundingbox != None):
          axialtumorbounds[tumorboundingbox[2]]       = True
        datamatrix ['axialliverbounds']               = axialliverbounds
        datamatrix ['axialtumorbounds']               = axialtumorbounds
        datamatrix ['imagepath']                      = np.repeat(imagelocation, nslice )
        datamatrix ['truthpath']                      = np.repeat(truthlocation, nslice )
        datamatrix ['imagedata']                      = resimage.transpose(2,1,0)
        datamatrix ['truthdata']                      = restruth.transpose(2,1,0)
        datamatrix ['image_dx']                       = np.repeat( 1.0/( float(imagedata.header['pixdim'][1])*(_globalexpectedpixel / options.trainingresample)), nslice)
        datamatrix ['image_dy']                       = np.repeat( 1.0/( float(imagedata.header['pixdim'][2])*(_globalexpectedpixel / options.trainingresample)), nslice)
        datamatrix ['image_dz']                       = np.repeat( 1.0/( float(imagedata.header['pixdim'][3])), nslice)
        numpydatabase = np.hstack((numpydatabase,datamatrix))
        # count total slice for QA
        totalnslice = totalnslice + nslice
      else:
        print('training data error image[2] = %d , truth[2] = %d ' % (nslice,restruth.shape[2]))

  # save numpy array to disk
  np.save( _globalnpfile, numpydatabase)


##########################
# build NN model from anonymized data
##########################
def TrainModel(kfolds=options.kfolds,idfold=0):

  global _num_classes

  ###
  ### load data
  ###

  print('loading memory map db for large dataset')
  numpydatabase = np.load(_globalnpfile)
  (train_index,test_index) = GetSetupKfolds(kfolds,idfold)

  print('copy data subsets into memory...')
  axialbounds = numpydatabase['axialliverbounds']
  dataidarray = numpydatabase['dataid']
  dbtrainindex= np.isin(dataidarray, train_index )
  dbtestindex = np.isin(dataidarray, test_index  )
  subsetidx_train  = np.all( np.vstack((axialbounds , dbtrainindex)) , axis=0 )
  subsetidx_test   = np.all( np.vstack((axialbounds , dbtestindex )) , axis=0 )
  if np.sum(subsetidx_train) + np.sum(subsetidx_test) != min(np.sum(axialbounds ),np.sum(dbtrainindex )) :
      raise("data error: slice numbers dont match")

  print('copy memory map from disk to RAM...')
  trainingsubset = numpydatabase[subsetidx_train]

  np.random.seed(seed=0)
  np.random.shuffle(trainingsubset)

  totnslice = len(trainingsubset)

  x_train=trainingsubset['imagedata']
  y_train=trainingsubset['truthdata']

  x_train_dx = trainingsubset['image_dx']
  x_train_dy = trainingsubset['image_dy']
  x_train_dz = trainingsubset['image_dz']
  x_train_dims = np.transpose(np.vstack((x_train_dx, x_train_dy, x_train_dz)))

  slicesplit        = int(0.9 * totnslice)

  TRAINING_SLICES   = slice(         0, slicesplit)
  VALIDATION_SLICES = slice(slicesplit, totnslice )

  print("\nkfolds : ", kfolds)
  print("idfold : ", idfold)
  print("slices total      : ", totnslice)
  print("slices training   : ", slicesplit)
  print("slices validation : ", totnslice - slicesplit)
  print("slices holdout    : ", len(numpydatabase[subsetidx_test]), "\n")

  # Convert to uint8 data and find out how many labels.
  y_train_typed = y_train.astype(np.uint8)
  _num_classes = 1
  y_train_one_hot = y_train_typed[...,np.newaxis]
  y_train_one_hot[ y_train_one_hot > 0 ] = 1  # map all nonzero values to 1

  ###
  ### set up output, logging, and callbacks
  ###

  logfileoutputdir= '%s/%03d/%03d' % (options.outdir, kfolds, idfold)
  os.system ('mkdir -p %s' % logfileoutputdir)
  print("Output to\t", logfileoutputdir)

  if options.with_hvd:
      callbacks = [ hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                    hvd.callbacks.MetricAverageCallback(),
                    hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1),
                    keras.callbacks.TerminateOnNaN()         ]
      if hvd.rank() == 0:
          callbacks += [ keras.callbacks.ModelCheckpoint(filepath=logfileoutputdir+"/tumormodelunet.h5", verbose=1, save_best_only=True),
                         keras.callbacks.TensorBoard(log_dir=logfileoutputdir, histogram_freq=0, write_graph=True, write_images=False)      ]
  else:
      callbacks = [ keras.callbacks.TerminateOnNaN(),
                    keras.callbacks.ModelCheckpoint(filepath=logfileoutputdir+"/tumormodelunet.h5", verbose=1, save_best_only=True),  
                    keras.callbacks.TensorBoard(log_dir=logfileoutputdir, histogram_freq=0, write_graph=True, write_images=False)  ] 


  ###
  ### create and run model
  ###

  # initial values for u
  if options.randinit:
      x_init = np.random.uniform(size=(totnslice,_nx,_ny))
  elif options.circleinit:
      x_center = int(_ny*0.25)
      y_center = int(_nx*0.66)
      rad      = 30
      x_init = np.zeros((totnslice,_nx,_ny))
      x_init[:, y_center-rad:y_center+rad, x_center-rad:x_center+rad] = 1.0
  else:
      x_init = np.ones((totnslice,_nx,_ny))

  # set up optimizer
  if options.with_hvd:
      if options.trainingsolver=="adam":
          opt = keras.optimizers.Adam(lr=0.001*hvd.size())
      elif options.trainingsolver=="adadelta":
          opt = keras.optimizers.Adadelta(1.0*hvd.size())
      elif options.trainingsolver=="nadam":
          opt = keras.optimizers.Nadam(0.002*hvd.size())
      elif options.trainingsolver=="sgd":
          opt = Keras.optimizers.SGD(0.01*hvd*size())
      else:
          raise Exception("horovod-enabled optimizer not selected")
      opt = hvd.DistributedOptimizer(opt)
  else:
      opt = options.trainingsolver

  # compile model graph
  model = get_upwind_transport_net(_nt)
  model.compile(loss=dsc_as_l2,
        metrics=[dice_metric_zero],
        optimizer=opt)
  print("Model parameters: {0:,}".format(model.count_params()))
  print("Input image shape: ", x_train[TRAINING_SLICES,:,:,np.newaxis].shape)
  x_in  = np.concatenate((x_train[TRAINING_SLICES  , :,:,np.newaxis], x_init[TRAINING_SLICES,   :,:,np.newaxis]), axis=-1)
  x_val = np.concatenate((x_train[VALIDATION_SLICES, :,:,np.newaxis], x_init[VALIDATION_SLICES, :,:,np.newaxis]), axis=-1)
  
  model.summary()
  
  history = model.fit([    x_in, 
                           x_train_dims[TRAINING_SLICES,:,np.newaxis,np.newaxis] ], 
                       y_train_one_hot[TRAINING_SLICES ],
                       validation_data=([x_val, x_train_dims[VALIDATION_SLICES,:,np.newaxis,np.newaxis] ], y_train_one_hot[VALIDATION_SLICES]),
                       callbacks = callbacks,
                       batch_size=options.trainingbatch,
                       epochs=options.numepochs)

  ###
  ### make predicions on validation set
  ###

  validationimgnii     = nib.Nifti1Image(x_train[VALIDATION_SLICES,:,:] , None )
  validationonehotnii  = nib.Nifti1Image(y_train[VALIDATION_SLICES,:,:] , None )
  y_predicted          = model.predict( [x_val, x_train_dims[VALIDATION_SLICES,:,np.newaxis,np.newaxis] ])
  y_segmentation       = np.argmax(y_predicted , axis=-1)
  validationoutput     = nib.Nifti1Image( y_segmentation[:,:,:].astype(np.uint8), None )
#  for jjj in range(_num_classes):
  for jjj in range(1):
      validationprediction  = nib.Nifti1Image(y_predicted [:,:,:,jjj] , None )
      validationprediction.to_filename( '%s/validationpredict-%d.nii.gz' % (logfileoutputdir,jjj) )
  validationimgnii.to_filename(    '%s/validationimg.nii.gz'    % logfileoutputdir )
  validationonehotnii.to_filename( '%s/validationseg.nii.gz'    % logfileoutputdir )
  validationoutput.to_filename(    '%s/validationoutput.nii.gz' % logfileoutputdir )

  modelloc = "%s/tumormodelunet.h5" % logfileoutputdir
  return modelloc


##########################
# apply model to test set
##########################
def MakeStatsScript(kfolds=options.kfolds, idfold=0):
  databaseinfo = GetDataDictionary()
  maketargetlist = []
  # open makefile
  with open('kfold%03d-%03d-stats.makefile' % (kfolds, idfold), 'w') as fileHandle:
      (train_set,test_set) = GetSetupKfolds(kfolds, idfold)
      for idtest in test_set:
         uidoutputdir= '%s/%03d/%03d' % (options.outdir, kfolds, idfold)
         segmaketarget  = '%s/label-%04d.nii.gz' % (uidoutputdir,idtest)
         segmaketarget0 = '%s/label-%04d-0.nii.gz' % (uidoutputdir,idtest)
         segmaketargetQ = '%s/label-%04d-?.nii.gz' % (uidoutputdir,idtest)
         predicttarget  = '%s/label-%04d-all.nii.gz' % (uidoutputdir,idtest)
         statstarget    = '%s/stats-%04d.txt' % (uidoutputdir,idtest)
         maketargetlist.append(segmaketarget )
         imageprereq = '$(TRAININGROOT)/%s' % databaseinfo[idtest]['image']
         segprereq   = '$(TRAININGROOT)/%s' % databaseinfo[idtest]['label']
         votecmd = "c3d %s -vote -type uchar -o %s" % (segmaketargetQ, predicttarget)
         infocmd = "c3d %s -info > %s" % (segmaketarget0,statstarget)
         statcmd = "c3d -verbose %s %s -overlap 0 -overlap 1 > %s" % (predicttarget, segprereq, statstarget)
         fileHandle.write('%s: %s\n' % (segmaketarget ,imageprereq ) )
         fileHandle.write('\t%s\n' % votecmd)
         fileHandle.write('\t%s\n' % infocmd)
         fileHandle.write('\t%s\n' % statcmd)
    # build job list
  with open('kfold%03d-%03d-stats.makefile' % (kfolds, idfold), 'r') as original: datastream = original.read()
  with open('kfold%03d-%03d-stats.makefile' % (kfolds, idfold), 'w') as modified: modified.write( 'TRAININGROOT=%s\n' % options.rootlocation + "cvtest: %s \n" % ' '.join(maketargetlist) + datastream)

##########################
# apply model to new data
##########################
def PredictModel(model=options.predictmodel, image=options.predictimage, outdir=options.segmentation):
  if (model != None and image != None and outdir != None ):
  
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    
    imagepredict = nib.load(image)
    imageheader  = imagepredict.header
    numpypredict = imagepredict.get_data().astype(IMG_DTYPE )
    # error check
    assert numpypredict.shape[0:2] == (_globalexpectedpixel,_globalexpectedpixel)
    nslice = numpypredict.shape[2]
    print(nslice)
    resizepredict = skimage.transform.resize(numpypredict,
            (options.trainingresample,options.trainingresample,nslice ),
            order=0,
            preserve_range=True,
            mode='constant').astype(IMG_DTYPE).transpose(2,1,0)

    # init conditions
    inshape = resizepredict.shape
    if options.randinit:
        inits = np.random.uniform(size=inshape)
    else:
        inits = np.ones(inshape)
#    x_center = int(inshape[2]*0.25)
#    y_center = int(inshape[1]*0.66)
#    rad      = 30
#    x_init = np.zeros(inshape)
#    x_init[:, y_center-rad:y_center+rad, x_center-rad:x_center+rad] = 1.0

    # set up optimizer
    if options.with_hvd:
        if options.trainingsolver=="adam":
            opt = keras.optimizers.Adam(lr=0.001*hvd.size())
        elif options.trainingsolver=="adadelta":
            opt = keras.optimizers.Adadelta(1.0*hvd.size())
        elif options.trainingsolver=="nadam":
            opt = keras.optimizers.Nadam(0.002*hvd.size())
        elif options.trainingsolver=="sgd":
            opt = Keras.optimizers.SGD(0.01*hvd*size())
        else:
            raise Exception("horovod-enabled optimizer not selected")
        opt = hvd.DistributedOptimizer(opt)
    else:
        opt = options.trainingsolver

    loaded_model = get_upwind_transport_net(_nt)
    loaded_model.compile(loss=dsc_as_l2,
          metrics=[dice_metric_zero],
          optimizer=opt)
    loaded_model.load_weights(model)
    print("Loaded model from disk")

    x_in = np.concatenate((resizepredict[...,np.newaxis], inits[...,np.newaxis]), axis=-1)
    x_dx = np.repeat( float(imageheader['pixdim'][1])*(_globalexpectedpixel / options.trainingresample), nslice)
    x_dy = np.repeat( float(imageheader['pixdim'][2])*(_globalexpectedpixel / options.trainingresample), nslice)
    x_dz = np.repeat( float(imageheader['pixdim'][3]), nslice)
    x_dims = np.transpose(np.vstack((x_dx, x_dy, x_dz)))
  
    segout = loaded_model.predict([x_in, x_dims[...,np.newaxis,np.newaxis]] )
    segout_resize = skimage.transform.resize(segout[...,0],
            (nslice,_globalexpectedpixel,_globalexpectedpixel),
            order=0,
            preserve_range=True,
            mode='constant').transpose(2,1,0)
    segout_img = nib.Nifti1Image(segout_resize, None, header=imageheader)
    segout_img.to_filename( outdir.replace('.nii.gz', '-%d.nii.gz' % 0) )

################################
# Perform K-fold validation
################################
def OneKfold(k=options.kfolds, i=0, datadict=None):
    mdlloc = TrainModel(kfolds=k, idfold=i) 
    (train_set,test_set) = GetSetupKfolds(k,i)
    for idtest in test_set:
        baseloc = '%s/%03d/%03d' % (options.outdir, k, i)
        imgloc  = '%s/%s' % (options.rootlocation, datadict[idtest]['image'])
        outloc  = '%s/label-%04d.nii.gz' % (baseloc, idtest) 
        if options.numepochs > 0:
            PredictModel(model=mdlloc, image=imgloc, outdir=outloc )
    MakeStatsScript(kfolds=k, idfold=i)

def Kfold(kkk):
    databaseinfo = GetDataDictionary()
    for iii in range(kkk):
        OneKfold(k=kkk, i=iii, datadict=databaseinfo)


if options.builddb:
    BuildDB()
if options.kfolds > 1:
    if options.idfold > -1:
        databaseinfo = GetDataDictionary()
        OneKfold(k=options.kfolds, i=options.idfold, datadict=databaseinfo)
    else:
        Kfold(options.kfolds)
if options.trainmodel: # no kfolds, i.e. k=1
    TrainModel(kfolds=1,idfold=0)
if options.predictmodel:
    PredictModel()
if ( (not options.builddb) and (not options.trainmodel) and (not options.predictmodel) and (options.kfolds == 1)):
    parser.print_help()