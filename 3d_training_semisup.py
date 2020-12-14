#!/usr/bin/env python
# coding: utf-8

# voxel morph imports
import tensorflow as tf
import voxelmorph as vxm
import neurite as ne
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'

import numpy as np
import h5py
import os

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

label_vals = np.array([1,2,3,4,5,6,7])

# build vxm network
vxm_model = vxm.networks.VxmDenseSemiSupervisedSeg(inshape=vol_shape,
                                                   nb_labels=len(label_vals),
                                                   nb_unet_features=nb_features,
                                                   int_steps=0);

# voxelmorph has a variety of custom loss classes
losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

vols_names = ['vol'+str(i)+'.npz' for i in keys_train]

# usually, we have to balance the two losses by a hyper-parameter
lambda_param = 0.05
loss_weights = [1, lambda_param]

batch_size = 4

vxm_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
				  loss=losses,
				  loss_weights=loss_weights)

#train generator
train_generator = vxm.generators.semisupervised(vol_names=vols_names[31:],
                                                atlas_file='vol904.npz',
                                                labels=label_vals
                                               )

#test generator
val_generator = vxm.generators.semisupervised(vol_names=vols_names[:31],
                                              atlas_file='vol904.npz',
                                              labels=label_vals
                                              )

val_entries = 30
tmp = [next(val_generator) for i in range(val_entries)]
x = [
    [tmp[i][0][0] for i in range(val_entries)],
    [tmp[i][0][1] for i in range(val_entries)],
    [tmp[i][0][2] for i in range(val_entries)]
    ]

y = [
    [tmp[i][1][0] for i in range(val_entries)],
    [tmp[i][1][1] for i in range(val_entries)],
    [tmp[i][1][2] for i in range(val_entries)]
	]

xy_val = (x,y)

hist = vxm_model.fit(train_generator,
                     validation_data=xy_val,
                     validation_batch_size=batch_size,
                     epochs=32,
                     steps_per_epoch=250,
                     verbose=1)

vxm_model.save_weights("wght_3d_semisup_val_labels7_"+str(lambda_param)+".keras")

util.export_history(hist, "hist_3d_semisup_val_labels7_"+str(lambda_param)+".txt")
