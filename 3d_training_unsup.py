#!/usr/bin/env python
# coding: utf-8

# voxel morph imports
import tensorflow as tf
import voxelmorph as vxm
import neurite as ne
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'

import numpy as np
import h5py

import utilitary as util

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


# voxelmorph has a variety of custom loss classes
losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

atlas = np.load('vol904.npz')['vol'][np.newaxis,...,np.newaxis]

vols_names = ['vol'+str(i)+'.npz' for i in keys_train]

# usually, we have to balance the two losses by a hyper-parameter
lambda_param = 0.05
loss_weights = [1, lambda_param]

batch_size = 4

vxm_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
				  loss=losses,
				  loss_weights=loss_weights)

# create a generator with a constant fixed volume (keys='853')
#chose this one because middle of the movie and provided with a label mask
train_generator = vxm.generators.scan_to_atlas(vols_names[31:],
											   atlas = atlas,
											   batch_size=batch_size)

slices_val_3d = np.zeros((len(keys_train[:31]),112,112,32))
for i, key in enumerate(keys_train[:31]):
    slices_val_3d[i] = np.load('vol'+key+'npz')['vol']

xy_val = util.create_xy_3d(slices_val_3d, atlas.squeeze())

hist = vxm_model.fit(train_generator,
                     validation_data=xy_val,
                     validation_batch_size=batch_size,
                     epochs=3,
                     steps_per_epoch=2,
                     verbose=1,
                     workers=4)

vxm_model.save_weights("wght_3d_unsup_"+str(lambda_param)+".keras")

util.export_history(hist, "hist_unsup_"+str(lambda_param)+".txt")
