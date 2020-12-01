# imports
import os, sys

#vxm
import numpy as np
import tensorflow as tf
import voxelmorph as vxm
import neurite as ne
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'

#dataset
import h5py

#utilitary
import utilitary as util

#plotting
import matplotlib.pyplot as plt

#preprocessing
from scipy import ndimage
import cv2

#import dataset
hf = h5py.File("dataset2.h5", "r")

keys = hf.keys()
use_key = util.remove_empty_key(keys)
x,y,z = hf.get('0')['frame'][0].shape
       
nb_entries = len(use_key)
list_keys = list(use_key)

# Split train-validation set
# 80% train - 20% validation
ratio = 0.8
keys_random = np.random.permutation(nb_entries)
keys_train = keys_random[:int(nb_entries * ratio)]
keys_test  = keys_random[int(nb_entries * ratio):]

#projection of red canal on z plan
r_MIP = util.red_MIP(hf, list_keys, axis = 2)

#contour detection + cropping center of mass
slices_train = np.array([util.crop_ctr_mass(r_MIP[i]) for i in keys_train])
slices_test  = np.array([util.crop_ctr_mass(r_MIP[i]) for i in keys_test])

# UNET architecture

vol_shape = (x, y) # 32 slices
nb_features = [
    [16, 32, 32, 32],             # encoder
    [32, 32, 32, 32, 32, 16, 16]  # decoder
]

# build vxm network using VxmDense
vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0);

# Losses : MSE + smoothness (regularization) 
losses = ['mse', vxm.losses.Grad('l2').loss]

# Regularizer
lambdas = np.logspace(-4,0,4)
for lambda_ in lambdas :

    loss_weights = [1, lambda_]
    
    #vxm_model = tf.keras.utils.multi_gpu_model(vxm_model, gpus=2)
    # Adam optimizer learning rate
    vxm_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=losses, loss_weights=loss_weights)
    
    # Training
    train_generator = util.vxm_data_generator(slices_train, batch_size = 1)
    hist = vxm_model.fit(train_generator, epochs=9, steps_per_epoch= 50, verbose=1);
    # Visualize the losses
    plot_history(hist, save_name = str(lambda_))
    title = "weights" + str(lambda_) + ".keras"
    # If it looks ok => save the weights
    vxm_model.save_weights(title)
    
# Load the best weights found
title = "weights" + str(0.001) + ".keras"
vxm_model.load_weights(title)

# Validation set generator
val_generator = vxm_data_generator(slices_test, batch_size = 15)
val_input, _ = next(val_generator)
our_val_pred = vxm_model.predict(val_input);

moving = np.squeeze(val_input[0])
fixed = np.squeeze(val_input[1])
moved = np.squeeze(our_val_pred[0])

a = [moved[i, ...] for i in range(5)]
b = [moving[i, ...] for i in range(5)]
c = [fixed[i, ...] for i in range(5)]

# Moving
ne.plot.slices(b, do_colorbars=True);
# Moved
ne.plot.slices(a, do_colorbars=True);
# Fixed
ne.plot.slices(c, do_colorbars=True);