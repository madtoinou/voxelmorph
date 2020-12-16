#!/usr/bin/env python
# coding: utf-8
"""
    python 2d_training.py --image-loss 'mse' --grad-loss-weight 0.005 --savename 'name' --epochs 1 --steps-per-epoch 90
"""
# voxel morph imports
import tensorflow as tf
import voxelmorph as vxm
import neurite as ne
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'
import argparse
import numpy as np
import scipy.ndimage as ndi
import h5py
import cv2
import os
import utilitary as util

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--savename', help='saving name')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID numbers (default: 0)')
parser.add_argument('--epochs', type=int, default=90, help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=250, help='frequency of model saves (default: 100)')
parser.add_argument('--batch-size', type=int, default=16, help='size of the batch')
# parser.add_argument('--load-weights', help='optional weights file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--seed', type=int, default=336699, help='random seed')
parser.add_argument('--nb-fixed', type=int, default=0, help='number of the fixed entry')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+', help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7, help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=1, help='flow downsample factor for integration (default: 2)')

# loss hyperparameters
parser.add_argument('--image-loss', default='mse', help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--grad-loss-weight', type=float, default=0.01, help='weight of gradient loss (lamba) (default: 0.01)')
args = parser.parse_args()



np.random.seed(args.seed)
#load data
hf = h5py.File("dataset2.h5", "r")
cropped_img = np.load('crop_r.npy')
keys = hf.keys()
use_key = util.remove_empty_key(hf, keys)
# removes entries with 1 contour (impossible to rotate atm)
del use_key[184]
del use_key[459]
nb_entries = len(use_key)
x,y,z = hf.get('0')['frame'][0].shape
# the data are cropped, compared to the original
(x,y) = (int(x/2),int(y/2))

max_r = 1543
min_r = 0
max_g = 1760
min_g = 0

# Split train-validation set
# 90% train  - 10% validation - 10% test
key_fixed = args.nb_fixed
keys_random = np.random.permutation(nb_entries)
# remove the fixed key from the keys
keys_random = keys_random[keys_random != key_fixed]
keys_train = keys_random[:int(nb_entries * 0.8)]
keys_val = keys_random[int(nb_entries * 0.8):int(nb_entries * 0.9)]

# our data will be of shape 256 x 256
vol_shape = (x,y)
# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

# build vxm network
vxm_model = vxm.networks.VxmDense(inshape = vol_shape,
                                  nb_unet_features=[enc_nf, dec_nf],
                                  int_steps=0);

# prepare image loss
# normalized cross-correlation
if args.image_loss == 'ncc':
    image_loss_func = vxm.losses.NCC().loss
# mean-squared error
elif args.image_loss == 'mse':
    image_loss_func = vxm.losses.MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)


# losses
losses  = [image_loss_func, vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
weights = [1, args.grad_loss_weight]

#load the training set + normalization
slices_train_r = np.zeros((len(keys_train),x,y))
for i, key in enumerate(keys_train):
    slices_train_r[i] = (cropped_img[key]-min_r)/(max_r-min_r)

#load the validation set + normalization
slices_val_r = np.zeros((len(keys_val),x,y))
for i, key in enumerate(keys_val):
    slices_val_r[i] = (cropped_img[key]-min_r)/(max_r-min_r)

# fixed/reference image
slice_fixed = (cropped_img[key_fixed]-min_r)/(max_r-min_r)
# validation set
xy_val = util.create_xy_3d(slices_val_r, fixed_vol= slice_fixed)
# create a Training generator with a constant fixed
train_generator = util.vxm_data_generator(slices_train_r,
                                          vol_fixed= slice_fixed,
                                          batch_size=args.batch_size)


vxm_model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr), loss=losses, loss_weights=weights)

hist = vxm_model.fit(train_generator,
                     validation_data=xy_val,
                     validation_batch_size=50,
                     epochs=args.epochs,
                     steps_per_epoch=slices_train_r.shape[0]//args.batch_size,
                     verbose=1)

vxm_model.save_weights(args.savename+str(args.grad_loss_weight)+".keras")

util.export_history(hist, args.savename+str(args.grad_loss_weight)+".txt")
