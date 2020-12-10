#!/usr/bin/env python
# coding: utf-8

# imports
import os, sys

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

def plot_history(hist, loss_name='loss'):
    # Simple function to plot training history.
    plt.figure()
    plt.plot(hist.epoch, hist.history[loss_name], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

hf = h5py.File("epfl3.h5", "r")

nb_entries = len(hf.keys())
list_keys = list(hf.keys())
keys_random = np.random.permutation(list_keys)

keys_train = keys_random[:int(nb_entries*0.8)]
keys_test  = keys_random[int(nb_entries*0.8):]

"""
#load the training set + normalization
slices_train = np.zeros((len(keys_train),112,112,32))
for i, key in enumerate(keys_train):
    slices_train[i] = np.array(hf.get(key)["frame"][0][:,:,:])/255

#load the training set + normalization
slices_test = np.zeros((len(keys_test),112,112,32))
for i, key in enumerate(keys_test):
    slices_test[i] = np.array(hf.get(key)["frame"][0][:,:,:])/255

# extract some brains
idx = np.random.randint(0, len(slices_train), 25)
example_digits = [f for f in slices_train[idx, ...]]

MIP_example = util.np_MIP(slices_train, idx, 2)
plots = [MIP_example[i] for i in range (len(MIP_example))]

# visualize
ne.plot.slices(plots, cmaps=['gray'], do_colorbars=True,
              grid=[5,5]);

# let's test it
train_generator = util.vxm_data_generator(slices_train,
                                          vol_fixed=np.array(hf.get('853')["frame"][0][:,:,:])/255,
                                          batch_size=4)

hist = vxm_model.fit(train_generator, epochs=3, steps_per_epoch=5, verbose=1);

# as before, let's visualize what happened
plot_history(hist)

# create the validation data generator
val_generator = util.vxm_data_generator(slices_test,
                                          vol_fixed=np.array(hf.get('853')["frame"][0][:,:,:])/255,
                                          batch_size=4)

val_input, _ = next(val_generator)

# prediction
val_pred = vxm_model.predict(val_input)

images = [np.squeeze(img).shape for img in val_input]

val_input, _ = next(val_generator)

# prediction
val_pred = vxm_model.predict(val_input)

# visualize registration
images = [np.squeeze(util.np_MIP(np.squeeze(img),[0],2)) for img in val_input+[val_pred[0]]] 
titles = ['moving', 'fixed', 'moved', 'flow']
ne.plot.slices(images, titles=titles[:-1], cmaps=['gray'], do_colorbars=True);

vxm_model.save_weights("weights.keras")

# # Evaluation

# Evaluating registration results is tricky. The first tendancy is to look at the images (as above), and conclude that if they match, The registration has succeeded.
# 
# However, this can be achieved by an optimization that only penalizes the image matching term. For example, next we compare our model with one that was trained on maximizing MSE only (without smoothness loss).


# prediction from model with MSE + smoothness loss
vxm_model.load_weights('weights.keras')
our_val_pred = vxm_model.predict(val_input)

# prediction from model with just MSE loss
vxm_model.load_weights('weights.keras')
mse_val_pred = vxm_model.predict(val_input)


# visualize MSE + smoothness model output
images = [np.squeeze(util.np_MIP(np.squeeze(img),[0],2)) for img in [val_input[1], our_val_pred[0]]]
titles = ['fixed', 'MSE + smoothness', 'flow']
ne.plot.slices(images, titles=titles[:-1], cmaps=['gray'], do_colorbars=True);

# visualize MSE model output
images = [np.squeeze(util.np_MIP(np.squeeze(img),[0],2)) for img in [val_input[1], mse_val_pred[0]]]
titles = ['fixed', 'MSE only', 'flow']
ne.plot.slices(images, titles=titles[:-1], cmaps=['gray'], do_colorbars=True);
"""

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


val_input = [x[np.newaxis, ..., np.newaxis] for x in slices_train_3d]

# voxelmorph has a variety of custom loss classes
losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

# usually, we have to balance the two losses by a hyper-parameter
lambda_param = 0.05
loss_weights = [1, lambda_param]


vxm_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=losses, loss_weights=loss_weights)

train_generator = util.vxm_data_generator(slices_train_3d,
                                          vol_fixed=np.array(hf.get('853')["frame"][0][:,:,:])/255,
                                          batch_size=4)

vxm_model.fit(train_generator, epochs=5, steps_per_epoch=10, verbose=1);

vxm_model.save_weights("wght_3d_"+str(lambda_param)+".keras")