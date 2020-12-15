# voxelmorph: Learning-Based Image Registration  

Adapted for Professor Sahand LPBS Lab at EPFL by Ines, Loic and Antoine

# Tutorial

Visit the [VoxelMorph tutorial](http://tutorial.voxelmorph.net/) to learn about VoxelMorph and Learning-based Registration


# Instructions

To use the VoxelMorph library, either clone this repository and install the requirements listed in `setup.py` or install directly with pip.

```
pip install voxelmorph
```

## Before Training

Transform the data into npz.

```
python create_npz.py
```

## Training unsupervised
For '--image-loss' : 'mse' = mean-squared error or 'ncc' = normalized cross-correlation
'--grad-loss-weight' : the L2-regularizer

```
python 3d_training_unsup.py --image-loss 'mse' --grad-loss-weight 0.05 --savename 'name'
```

## Training semisupervised

```
python 3d_training_semipsup.py --image-loss 'mse' --grad-loss-weight 0.05 --savename 'name'
```
