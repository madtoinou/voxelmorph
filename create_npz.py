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
from numpy import savez_compressed
#plotting
import matplotlib.pyplot as plt

np.random.seed(336699)

hf = h5py.File("epfl3.h5", "r")

nb_entries = len(hf.keys())
list_keys = list(hf.keys())
keys_random = np.random.permutation(list_keys)

keys_train = keys_random[:int(nb_entries*0.8)]
keys_test  = keys_random[int(nb_entries*0.8):]

#get the index of keys with frame + labels
masked_train = [i for i in keys_train if (len(hf.get(i)) > 1)]

#preallocation
labels_train_3d_semisup = np.zeros((len(keys_train),112,112,32))
slices_train_3d_semisup = np.zeros((len(keys_train),112,112,32))

#load the training set + normalization
for i, key in enumerate(keys_train):
    if i in masked_train:
        labels_train_3d_semisup[i] = np.array(hf.get(key)["mask"])
        slices_train_3d_semisup[i] = np.array(hf.get(key)["frame"][0][:,:,:])/255
    else:
        labels_train_3d_semisup[i] = np.zeros((112, 112,  32))
        slices_train_3d_semisup[i] = np.array(hf.get(key)["frame"][0][:,:,:])/255
        
# generate a file for each semi-supervised training entry
for i in range(len(slices_train_3d_semisup)):
    savez_compressed('vol'+str(i)+'.npz',
                     vol=slices_train_3d_semisup[i],
                     seg=labels_train_3d_semisup[i])

#get the index of keys with frame + labels
masked_test = [i for i in keys_test if (len(hf.get(i)) > 1)]

#preallocation
labels_test_3d_semisup = np.zeros((len(keys_test),112,112,32))
slices_test_3d_semisup = np.zeros((len(keys_test),112,112,32))

#load the test set + normalization
for i, key in enumerate(keys_test[:51]):
    if i in masked_test:
        labels_test_3d_semisup[i] = np.array(hf.get(key)["mask"])
        slices_test_3d_semisup[i] = np.array(hf.get(key)["frame"][0][:,:,:])/255
    else:
        labels_test_3d_semisup[i] = np.zeros((112, 112,  32))
        slices_test_3d_semisup[i] = np.array(hf.get(key)["frame"][0][:,:,:])/255
        
# generate a file for each semi-supervised training entry
for i in range(len(slices_test_3d_semisup)):
    savez_compressed('val'+str(i)+'.npz',
                     vol=slices_test_3d_semisup[i],
                     seg=labels_test_3d_semisup[i])
