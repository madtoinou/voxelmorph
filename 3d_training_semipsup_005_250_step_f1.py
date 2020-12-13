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

fixed_vol = np.array(hf.get('1')["frame"][0][:,:,:])/255

del hf

### 3D MOTHERF*

# our data will be of shape 112 x 112 x 32
vol_shape = (112, 112, 32)
nb_features = [
    [16, 32, 32, 32],
    [32, 32, 32, 32, 32, 16, 16]
]

# voxelmorph has a variety of custom loss classes
losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

# usually, we have to balance the two losses by a hyper-parameter
lambda_param = 0.05
loss_weights = [1, lambda_param]

batch_size = 4

# declare the model, using all 7 labels values
vxm_model_semisup = vxm.networks.VxmDenseSemiSupervisedSeg(inshape=vol_shape,
                                                           nb_labels=7,
                                                           nb_unet_features=nb_features,
                                                           int_steps=0);

vxm_model_semisup.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                          loss=losses,
                          loss_weights=loss_weights)

#nb_entries
l_train = len(keys_train)
vols_names = ['vol'+str(i)+'.npz' for i in range(l_train)]

l_test = len(keys_test)
vals_names = ['val'+str(i)+'.npz' for i in range(l_test)]

label_vals = np.array([1,2,3,4,5,6,7])
train_generator = vxm.generators.semisupervised(vol_names=vols_names,
                                                labels=label_vals
                                               )

hist = vxm_model_semisup.fit(train_generator,
                             epochs=32,
                             steps_per_epoch=250,
                             verbose=1);

vxm_model_semisup.save_weights("wght_3d_semisup_f1_250step_"+str(lambda_param)+".keras")

util.export_history(hist, "hist_3d_semisup_f1_250step_"+str(lambda_param)+".txt")
