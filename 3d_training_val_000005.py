#!/usr/bin/env python
# coding: utf-8

# voxel morph imports
import tensorflow as tf
import voxelmorph as vxm
import neurite as ne
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'

import numpy as np
import scipy.ndimage as ndi
import h5py
import cv2

import utilitary as util

#plotting
import matplotlib.pyplot as plt

np.random.seed(336699)

hf = h5py.File("epfl3.h5", "r")

nb_entries = len(hf.keys())
list_keys = list(hf.keys())
keys_random = np.random.permutation(list_keys)

keys_train = keys_random[:int(nb_entries*0.8)]
keys_test  = keys_random[int(nb_entries*0.8):]

### 3D MOTHERF*

# our data will be of shape 112 x 112 x 32
vol_shape = (112, 112, 32)
nb_features = [
    [16, 32, 32, 32],
    [32, 32, 32, 32, 32, 16, 16]
]

# build vxm network
vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0);

#load the training set + normalization
slices_train_3d = np.zeros((len(keys_train),112,112,32))
for i, key in enumerate(keys_train):
    slices_train_3d[i] = np.array(hf.get(key)["frame"][0][:,:,:])/255

#load the training set + normalization
slices_test_3d = np.zeros((len(keys_test),112,112,32))
for i, key in enumerate(keys_test):
    slices_test_3d[i] = np.array(hf.get(key)["frame"][0][:,:,:])/255

# voxelmorph has a variety of custom loss classes
losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

fixed_vol = np.array(hf.get('853')["frame"][0][:,:,:])/255

# usually, we have to balance the two losses by a hyper-parameter
lambda_param = 0.00005
loss_weights = [1, lambda_param]

batch_size = 4 

vxm_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=losses, loss_weights=loss_weights)

# create a generator with a constant fixed volume (keys='853')
#chose this one because middle of the movie and provided with a label mask
train_generator = util.vxm_data_generator(slices_train_3d[100:],
                                          vol_fixed=fixed_vol,
                                          batch_size=batch_size)

xy_val = util.create_xy_3d(slices_train_3d[:100], fixed_vol)

checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model3D_MSE" + str(lambda_param) + ".hdf5", monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='auto', period=1)
hist = vxm_model.fit(train_generator,
                     validation_data=xy_val,
                     validation_batch_size=batch_size,
                     epochs=32,
                     steps_per_epoch=32,
                     verbose=0, callbacks=[checkpoint])

vxm_model.save_weights("wght_3d_MSE_val_"+str(lambda_param)+".keras")

util.export_history(hist, "hist_3d_MSE_val_"+str(lambda_param)+".txt")