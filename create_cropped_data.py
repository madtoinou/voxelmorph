# imports
import os, sys
import numpy as np
import argparse
import h5py
import cv2
import utilitary as util

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=336699, help='random seed')
parser.add_argument('--thresh', type=int, default=130, help='binary threshold')
args = parser.parse_args()



np.random.seed(args.seed)
# load the raw data
hf = h5py.File("dataset2.h5", "r")
keys = hf.keys()
# the last three entries are empty
use_key = util.remove_empty_key(hf, keys)
# removes entries with 1 contour (impossible to rotate atm)
del use_key[184]
del use_key[459]
# Get the shape of a frame
x,y,z = hf.get('0')['frame'][0].shape
nb_entries = len(use_key)

# Find the mask of every frame of the red channel and apply it on the green channel
g_masked = np.empty((len(use_key), x, y, z), dtype=np.float32)

for i, key in enumerate(use_key):
    red = hf.get(key)['frame'][0]
    red = red.astype('float32')
    # median blur to remove the salt-and-pepper noise
    red = cv2.medianBlur(red, 5)
    # binary threshold, with threshold = 130
    mask_red = red > args.thresh

    # apply it on the green channel
    green = hf.get(key)['frame'][1]
    green = green.astype('float32')
    green = cv2.medianBlur(green, 5)
    g_masked[i,...] = green * mask_red

# MIP the red and the green channel, along z axis
r_MIP, _ = util.MIP_GR(hf, use_key, axis = 2)
g_MIP = util.np_MIP(g_masked, use_key, axis = 2)

# contours of the red channel
MIP_ctr, _, ctr_list = util.find_contour(r_MIP, blur=5)

#rotated image and rotated contour
rotated_img = [
    util.rot_img(r_MIP[i],
                 MIP_ctr[i],
                 ctr_list[i]) for i in range(len(use_key))
    ]

rotated_gr = [
    util.rot_img(g_MIP[i],
                 MIP_ctr[i],
                 ctr_list[i]) for i in range(len(use_key))
    ]

#contain the image cropped around the contour center of mass
cropped_img = [util.crop_ctr_mass(rotated_img[i]) for i in range(len(use_key))]
cropped_gre = [util.crop_ctr_mass(rotated_gr[i]) for i in range(len(use_key))]

# Save the cropped image for further use
np.save('crop_r.npy', cropped_img)
np.save('crop_g.npy', cropped_gre)
