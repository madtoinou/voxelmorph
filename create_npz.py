#!/usr/bin/env python
# coding: utf-8

import h5py

from numpy import savez_compressed
import numpy as np

hf = h5py.File("epfl3.h5", "r")

list_keys = list(hf.keys())

#export the training set + normalization
for key in list_keys:
    #clamping
    tmp=np.array(hf.get(key)["frame"][0])
    tmp[np.where(tmp > 255)] = 255
    
    #save normalize frame and its mask
    if len(hf.get(key)) > 1:
    	savez_compressed('vol'+str(key)+'.npz',
                     vol=(tmp-93)/(255-93),
                     seg=np.array(hf.get(key)["mask"]))
    else:
    	savez_compressed('vol'+str(key)+'.npz',
                     vol=(tmp-93)/(255-93),
                     seg=np.zeros((112, 112,  32)))