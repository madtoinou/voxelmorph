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

hf = h5py.File("../dataset2.h5", "r")
keys = hf.keys()
use_key = util.remove_empty_key(hf, keys)
# removes entries with 1 contour (impossible to rotate atm)
del use_key[184]
del use_key[459]
x,y,z = hf.get('0')['frame'][0].shape
(x,y) = (int(x/2),int(y/2) )
nb_entries = len(use_key)


max_r = 1543
min_r = 0
max_g = 1760
min_g = 0

# Split train-validation set
# 90% train  - 10% validation - 10% test
key_fixed = 0
keys_random = np.random.permutation(nb_entries)
keys_random = keys_random[keys_random != key_fixed]
keys_train = keys_random[:int(nb_entries * 0.8)]
keys_val = keys_random[int(nb_entries * 0.8):int(nb_entries * 0.9)]
keys_test  = keys_random[int(nb_entries * 0.9):]

### 2D MOTHERF*

# our data will be of shape 112 x 112
vol_shape = (x,y)
nb_features = [
    [16, 32, 32, 32],
    [32, 32, 32, 32, 32, 16, 16]
]

# build vxm network
vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0);

# load the cropped images of red
cropped_img = np.load('crop_r.npy')
#load the training set + normalization
slices_train_r = np.zeros((len(keys_train),x,y))
for i, key in enumerate(keys_train):
    slices_train_r[i] = (cropped_img[key]-min_r)/(max_r-min_r)

#load the validation set + normalization
slices_val_r = np.zeros((len(keys_val),x,y))
for i, key in enumerate(keys_val):
    slices_val_r[i] = (cropped_img[key]-min_r)/(max_r-min_r)

#load the training set + normalization
slices_test_r = np.zeros((len(keys_test),x,y))
for i, key in enumerate(keys_test):
    slices_test_r[i] = (cropped_img[key]-min_r)/(max_r-min_r)

# The first entry is the fixed
slice_fixed = (cropped_img[0]-min_r)/(max_r-min_r)

# vols_names = ['simu2d/vol'+str(i)+'.npz' for i in range(len(slices_train_r))]
# atlas_name = 'simu2d/atlas.npz'

# voxelmorph has a variety of custom loss classes
losses = [vxm.losses.NCC().loss, vxm.losses.Grad('l2').loss]

# usually, we have to balance the two losses by a hyper-parameter
lambdas_param = [1, 1.5]

for lambda_param in lambdas_param:

    loss_weights = [1, lambda_param]

    batch_size = 16

    vxm_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=losses, loss_weights=loss_weights)

    # validation. The last entry is the fixed
    xy_val = util.create_xy_3d(slices_val_r, fixed_vol = slice_fixed)
    # create a generator with a constant fixed idx : 0
    # Training
    train_generator = util.vxm_data_generator(slices_train_r,
                                              vol_fixed= slice_fixed,
                                              batch_size=batch_size)
    # train_generator = volgen(vol_names, batch_size=batch_size, return_segs=False, np_var='vol')

    checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model2D_NCC" + str(lambda_param) + ".hdf5", monitor='val_loss', verbose=1,
                                     save_best_only=True, mode='auto', period=1)
    hist = vxm_model.fit(train_generator,
                         validation_data=xy_val,
                         validation_batch_size=50,
                         epochs=90,
                         steps_per_epoch=slices_train_r.shape[0]//batch_size,
                         verbose=1)

    vxm_model.save_weights("wght_2d_NCC_"+str(lambda_param)+".keras")

    util.export_history(hist, "hist_2d_NCC_"+str(lambda_param)+".txt")
