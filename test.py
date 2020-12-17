"""
 python test.py --name-weights semisup_06_all_labels_ncc0.6.keras
"""

# voxel morph imports
import tensorflow as tf
import voxelmorph as vxm
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'
import argparse
import numpy as np
import h5py

import utilitary as util

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--atlas', default = '904', help='optional atlas to perform scan-to-atlas training')
parser.add_argument('--name-weights', help='the name of the model weights')
parser.add_argument('--seed', type=int, default=336699, help='random seed')
parser.add_argument('--model', default='semisup', help='unsup or semisup')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+', help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
args = parser.parse_args()

np.random.seed(args.seed)

#load the data-set
hf = h5py.File("epfl3.h5", "r")

#list of all the entries (the frames) from the movie
list_keys = list(hf.keys())
#The atlas should not be taken in the training or testing set
list_keys.remove(args.atlas)
nb_entries = len(hf.keys())
#Shuffle the keys before selecting training & testing keys
keys_random = np.random.permutation(list_keys)
keys_train = keys_random[:int(nb_entries*0.8)]
keys_test  = keys_random[int(nb_entries*0.8):]
# The reference/the atlas/the fixed is chosen
fixed_seg = np.array(hf.get(args.atlas)["mask"])

# keys in test set with labels mask
mask_tests = []
for i in keys_test:
    if len(hf.get(i)) > 1:
        mask_tests.append(i)

vol_shape = fixed_seg.shape
# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

# Different values possible of the labels
label_vals = np.array([1,2,3,4,5,6,7])
if args.model == 'semisup':
    vxm_model = vxm.networks.VxmDenseSemiSupervisedSeg(
                                            inshape=vol_shape,
                                            nb_labels=len(label_vals),
                                            nb_unet_features=[enc_nf, dec_nf],
                                            int_steps=0)
else :
    # build vxm network
    vxm_model = vxm.networks.VxmDense(inshape = vol_shape,
                                      nb_unet_features=[enc_nf, dec_nf],
                                      int_steps=0)

vxm_model.load_weights(args.name_weights)
#load test volumes
vols_names = ['vol'+str(i)+'.npz' for i in keys_test]

if args.model == 'semisup':

    predict_generator = vxm.generators.semisupervised(
                            vol_names=vols_names,
                            labels=label_vals)
    val_input = [next(predict_generator) for i in range(len(mask_tests))]
else :
    # atlas
    atlas = np.load('vol' + args.atlas + '.npz')['vol'][np.newaxis,...,np.newaxis]
    predict_generator = vxm.generators.scan_to_atlas(vols_names,
                                                    atlas = atlas)
    val_input = [next(predict_generator) for i in range(len(mask_tests))]

# predict the transformation
val_pred = []
for i in range(len(mask_tests)):
    val_pred.append(vxm_model.predict(val_input[i]))

warp_model = vxm.networks.Transform(vol_shape, interp_method='nearest')

#checking that it's the right prediction vector
assert (len(mask_tests) == len(val_pred))

#load the test set + normalization
slices_test_3d_mask = np.empty((len(mask_tests),*vol_shape))
for i, key in enumerate(mask_tests):
    slices_test_3d_mask[i] = np.array(hf.get(key)["mask"])

# predict the labels transformation (unseen data)
warped_seg = [warp_model.predict([slices_test_3d_mask[i][np.newaxis,...,np.newaxis], val_pred[i][1]]) for i in range(len(mask_tests))]

dice = [vxm.py.utils.dice(slices_test_3d_mask[i], fixed_seg, label_vals) for i in range(len(mask_tests))]
warped_seg = np.array(warped_seg)

dice_warp = [vxm.py.utils.dice(warped_seg.squeeze()[i], fixed_seg, label_vals) for i in range(len(mask_tests))]

dice = np.array(dice)
dice_warp = np.array(dice_warp)

#label of the brightest neuron
label_nb = 3
print('Before, label', str(label_nb+1), '\nmean dice :', dice[:,label_nb].mean(), '\nstd dice :',dice[:,label_nb].std())
print('***')
print('After, label', str(label_nb+1), '\nmean dice :', dice_warp[:,label_nb].mean(), '\nstd dice :', dice_warp[:,label_nb].std())
print('***')
print('Before, all labels', '\nmean dice :', dice.mean(), '\nstd dice :',dice.std())
print('***')
print('After, all labels', '\nmean dice :', dice_warp.mean(), '\nstd dice :',dice_warp.std())
print('***')


#normalized cross-correlation
#ground truth
gt = []
#vxm perf
vxm = []
for i in range(len(slices_test_3d_mask)):
    #rough prealign
    gt.append(np.corrcoef(slices_test_3d_mask[i].ravel(),
                    fixed_seg.ravel())[0,1])


    #after morphing
    vxm.append(np.corrcoef(slices_test_3d_mask[i].ravel(),
                    warped_seg[i].squeeze().ravel())[0,1])

print('Normalized cross-correlation')
print('Ground truth mean :', np.array(gt).mean(), '\nstd :', np.array(gt).std())
print('***')
print('Vxm mean :', np.array(vxm).mean(), '\nstd :', np.array(vxm).std())
