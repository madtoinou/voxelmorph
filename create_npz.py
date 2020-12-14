#!/usr/bin/env python
# coding: utf-8

import h5py

from numpy import savez_compressed
import numpy as np

np.random.seed(336699)

hf = h5py.File("epfl3.h5", "r")

nb_entries = len(hf.keys())
list_keys = list(hf.keys())
keys_random = np.random.permutation(list_keys)

keys_train = keys_random[:int(nb_entries*0.8)]
keys_test  = keys_random[int(nb_entries*0.8):]

#export the training set + normalization
for key in list_keys:
    #clamping
    tmp=np.array(hf.get(key)["frame"])
    tmp[np.where(tmp > 255)] = 255
    
    #save normalize frame and its mask
    if len(hf.get(key)) > 1:
    	savez_compressed('vol'+str(key)+'.npz',
                     vol=tmp/255,
                     seg=np.array(hf.get(key)["mask"]))
    else:
    	savez_compressed('vol'+str(key)+'.npz',
                     vol=tmp/255,
                     seg=np.zeros((112, 112,  32)))