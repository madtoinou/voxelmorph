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
import os
from numpy import savez_compressed

import utilitary as util

#plotting
import matplotlib.pyplot as plt

np.random.seed(336699)

hf = h5py.File("../epfl3.h5", "r")

nb_entries = len(hf.keys())
list_keys = list(hf.keys())
keys_random = np.random.permutation(list_keys)

keys_train = keys_random[:int(nb_entries*0.8)]
keys_test  = keys_random[int(nb_entries*0.8):]


os.makedirs('simu3d_STA', exist_ok = True)

vol_atlas = np.array(hf.get('1')["frame"][0][:,:,:])/255
seg_atlas = np.array(hf.get('1')["mask"])
savez_compressed('simu3d_STA/atlas.npz',
                     vol=vol_atlas,
                     seg= seg_atlas)

#preallocation, the last 100 are kept for validation 
slices_train_3d = np.zeros((len(keys_train[:-100]),112,112,32))
for i, key in enumerate(keys_train[:-100]):
    slices_train_3d[i] = np.array(hf.get(key)["frame"][0][:,:,:])/255
    savez_compressed('simu3d_STA/vol'+str(i)+'.npz',
                         vol=slices_train_3d[i])

#load the training set + normalization
slices_test_3d = np.zeros((len(keys_test),112,112,32))
for i, key in enumerate(keys_test):
    slices_test_3d[i] = np.array(hf.get(key)["frame"][0][:,:,:])/255
    savez_compressed('simu3d_STA/vol'+str(i)+'.npz',
                     vol=slices_test_3d[i])
