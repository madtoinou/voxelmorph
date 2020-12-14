#!/usr/bin/env python
# coding: utf-8

"""
python 3d_training_semipsup.py --image-loss 'mse' --grad-loss-weight 0.05 --savename 'name' --epochs 1 --steps-per-epoch 3
"""
# voxel morph imports
import os
import random
import argparse
import glob
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'
import voxelmorph as vxm
# load data
import h5py
import utilitary as util
# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
# parser.add_argument('datadir', help='base data directory')
# parser.add_argument("--labels", required=True, help='labels to use in dice loss')
# parser.add_argument('--model-dir', default='models', help='model output directory (default: models)')
parser.add_argument('--atlas', default = '904', help='optional atlas to perform scan-to-atlas training')
parser.add_argument('--savename', help='saving name')


# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID numbers (default: 0)')
parser.add_argument('--epochs', type=int, default=32, help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=250, help='frequency of model saves (default: 100)')
parser.add_argument('--batch-size', type=int, default=4, help='size of the batch')
# parser.add_argument('--load-weights', help='optional weights file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--seed', type=int, default=336699, help='random seed')


# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+', help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7, help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=1, help='flow downsample factor for integration (default: 2)')

# loss hyperparameters
parser.add_argument('--image-loss', default='mse', help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--grad-loss-weight', type=float, default=0.01, help='weight of gradient loss (lamba) (default: 0.01)')
parser.add_argument('--dice-loss-weight', type=float, default=0.01, help='weight of dice loss (gamma) (default: 0.01)')
args = parser.parse_args()



np.random.seed(args.seed)

hf = h5py.File("epfl3.h5", "r")
list_keys = list(hf.keys())
#The atlas should not be taken into the training set
list_keys.remove(args.atlas)
nb_entries = len(list_keys)
keys_random = np.random.permutation(list_keys)
#Training on 80% of the dataset
keys_train = keys_random[:int(nb_entries*0.8)]
del hf

# load and prepare training data
train_vol_names = ['vol'+str(key)+'.npz' for key in keys_train]
assert len(train_vol_names) > 0, 'Could not find any training data'

# list of labels
train_labels = np.array([1,2,3,4,5,6,7])

# size of validation set
val_entries = 30
# create a generator with an atlas
# The first 30 entries are kept for validation
train_generator = vxm.generators.semisupervised(vol_names=train_vol_names[(val_entries+1):],
                                                labels=train_labels,
                                                atlas_file='vol' + args.atlas + '.npz')
# validation generator with an atlas
# the validation is taken from the 30 first training data
val_generator = vxm.generators.semisupervised(vol_names=train_vol_names[:(val_entries+1)],
                                              atlas_file='vol' + args.atlas + '.npz',
                                              labels=train_labels
                                              )

# put the validation set in the correct shape
tmp = [next(val_generator) for i in range(val_entries)]
x = [tmp[i][0] for i in range(val_entries)]
y = [tmp[i][1] for i in range(val_entries)]
xy_val = (x,y)

# extract shape from sampled input
inshape = next(train_generator)[0][0].shape[1:-1]

# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

with tf.device(device):

    # build the model
    model = vxm.networks.VxmDenseSemiSupervisedSeg(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        nb_labels=len(train_labels),
        int_steps=args.int_steps,
        int_downsize=args.int_downsize
    )

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
    losses  = [image_loss_func, vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss, vxm.losses.Dice().loss]
    weights = [1, args.grad_loss_weight, args.dice_loss_weight]

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr), loss=losses, loss_weights=weights)

    hist = model.fit(train_generator,
        initial_epoch=args.initial_epoch,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_data=xy_val,
        validation_batch_size=args.batch_size,
        verbose=1
    )
    #save the last weights
    model.save_weights(args.savename + str(args.grad_loss_weight) + ".keras")
    #Save the losses in a txt file
    util.export_history(hist, args.savename + str(args.grad_loss_weight) + ".txt")
