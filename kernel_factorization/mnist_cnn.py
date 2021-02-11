# -*- coding: utf-8 -*-
"""
Convoluted Neural Network
"""

import numpy as np
import tensorflow as tf
import struct
import gzip
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, DepthwiseConv2D
from depthwise_notebook import depthwise_factorization

# functions ================================================================
def read_mnist_files(images_file, labels_file):
    #read image file
    with gzip.open(images_file) as f:
        magic, nimages = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        shape = (nimages, nrows, ncols)
        image_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    
    #read image label file
    with gzip.open(labels_file) as f:
        magic, nlabels = struct.unpack(">II", f.read(8))
        shape = (nlabels, 1)   
        label_data = np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    
    return image_data, label_data
# ==========================================================================


#Reading MNIST data from file 
train_images, train_labels = read_mnist_files('./MNIST_Dataset/train-images-idx3-ubyte.gz', './MNIST_Dataset/train-labels-idx1-ubyte.gz')
test_images, test_labels   = read_mnist_files('./MNIST_Dataset/t10k-images-idx3-ubyte.gz', './MNIST_Dataset/t10k-labels-idx1-ubyte.gz')
#
#
##Reshaping and Normalizing the images
n_train_images, nrows, ncol = np.shape(train_images)
train_data = np.reshape(train_images, newshape=(n_train_images, nrows*ncol))
train_data = StandardScaler().fit_transform(train_data)

n_test_images, nrows, ncol = np.shape(test_images)
test_data = np.reshape(test_images, newshape=(n_test_images, nrows*ncol))
test_data = StandardScaler().fit_transform(test_data)

train_images = train_images.reshape(n_train_images, nrows, ncol, 1)
test_images = test_images.reshape(n_test_images, nrows, ncol, 1)


print('train_images shape:', train_images.shape)
print('Number of images in train_images', train_images.shape[0])
print('Number of images in test_images', test_images.shape[0])
print(n_test_images)


# Creating Model and adding the layers
model = Sequential()
#model.add(DepthwiseConv2D(kernel_size=(5,5), padding='same', strides=(1, 1), activation='linear', input_shape=(nrows, ncol, 1), use_bias=False))
model.add(Conv2D(32, kernel_size=(5,5), padding='same', strides=(1, 1), activation='relu', input_shape=(nrows, ncol, 1)))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(DepthwiseConv2D(kernel_size=(5,5), padding='same', strides=(1, 1), activation='linear', input_shape=(nrows, ncol, 1), use_bias=False))
model.add(Conv2D(32, kernel_size=(1,1), padding='same', strides=(1, 1), activation='relu', input_shape=(nrows, ncol, 1)))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(10, activation='softmax'))


# Compiling and fitting the model 
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=train_images,y=train_labels, epochs=10)
model.summary()
tf.keras.models.save_model(model, 'mnist_depthwisemodel.h5')

# Evaluating the model 
model.evaluate(test_images, test_labels)


# convswap implementation ===========================================
model_new = Sequential() 
model = tf.keras.models.load_model('mnist_model.h5')
model.summary()
model.evaluate(test_images, test_labels)
layer_lst = [l for l in model.layers]

with tf.name_scope('my_scope'):
    for ii in range(len(layer_lst)):
        layer_type = type(layer_lst[ii]).__name__
        if ii+1 < len(layer_lst):
            layer_next = type(layer_lst[ii+1]).__name__
            print(layer_lst[ii].name)
        else: 
            layer_next = None
        
        if layer_type=="Conv2D" and layer_type!="MaxPooling2D" and layer_lst[ii].input_shape==layer_lst[ii].output_shape:
            conv, bias = layer_lst[ii].get_weights()  
            D, W, _, err, idx = depthwise_factorization(np.array(conv))
            print(err[idx-1])
            
            D = D[...,np.newaxis]
            W = W.T
            W = W[np.newaxis, np.newaxis, ...]
            
            model_new.add(DepthwiseConv2D(kernel_size=layer_lst[ii].kernel_size, 
                                    padding='same', 
                                    activation='linear',
                                    use_bias=False,
                                    weights=[D]))
            
            model_new.add(Conv2D(filters=32, 
                           kernel_size=(1,1), 
                           padding='same', 
                           activation='relu',
                           name='conv2dweight_'+str(ii),
                           weights=[W,bias]))
        
        else:
            if layer_type == "Conv2D":
                layer_config = Conv2D.get_config(layer_lst[ii])
                layer_temp = Conv2D.from_config(layer_config)
            elif layer_type == "MaxPooling2D":
                layer_config = MaxPooling2D.get_config(layer_lst[ii])
                layer_temp = MaxPooling2D.from_config(layer_config)        
            elif layer_type == "Dense":
                layer_config = Dense.get_config(layer_lst[ii])
                layer_temp = Dense.from_config(layer_config)
            elif layer_type == "Flatten":
                layer_temp = Flatten()
            else: 
                layer_config = tf.keras.layers.Layer.get_config(layer_lst[ii])
                layer_temp = tf.keras.layers.Layer.from_config(layer_config)
            
            layer_weight = layer_lst[ii].get_weights()
            layer_temp.build(layer_lst[ii].input_shape)
            if layer_weight: 
                layer_temp.set_weights(layer_weight)

            model_new.add(layer_temp)

    model_new.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])    
    model_new.summary()
    
    tf.keras.models.save_model(model_new, 'mnist_swap_model3.h5')


# Evaluating the new model 
model_new.evaluate(test_images, test_labels)

