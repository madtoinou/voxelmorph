# voxelmorph: Learning-Based Image Registration  

Adapted for Professor Sahand LPBS Lab at EPFL by Ines, Loic and Antoine

# Tutorial

Visit the [VoxelMorph tutorial](http://tutorial.voxelmorph.net/) to learn about VoxelMorph and Learning-based Registration


# Instructions

To use the VoxelMorph library, either clone this repository and install the requirements listed in `setup.py` (recommanded, contains dependencies for volume pre-processing and data-set import)

```
pip install -r setup.txt
```

or install directly with pip.

```
pip install voxelmorph
```

## Training

If you would like to train your own model, you will likely need to customize some of the data loading code in `voxelmorph/generators.py` for your own datasets and data formats. However, it is possible to run many of the example scripts out-of-the-box, assuming that you have a directory containing training data files in npz (numpy) format. It's assumed that each npz file in your data folder has a `vol` parameter, which points to the numpy image data to be registered, and an optional `seg` variable, which points to a corresponding discrete segmentation (for semi-supervised learning). It's also assumed that the shape of all image data in a directory is consistent.

For a given `/path/to/training/data`, the following script will train the dense network (described in MICCAI 2018 by default) using scan-to-scan registration. Model weights will be saved to a path specified by the `--model-dir` flag.

```
./scripts/tf/train.py /path/to/training/data --model-dir /path/to/models/output --gpu 0
```

Scan-to-atlas registration can be enabled by providing an atlas file with the `--atlas atlas.npz` command line flag. If you'd like to train using the original dense CVPR network (no diffeomorphism), use the `--int-steps 0` flag to specify no flow integration steps. Use the `--help` flag to inspect all of the command line options that can be used to fine-tune network architecture and training.


## Registration

If you simply want to register two images, you can use the `register.py` script with the desired model file. For example, if we have a model `model.h5` trained to register a subject (moving) to an atlas (fixed), we could run:

```
./scripts/tf/register.py --moving moving.nii.gz --fixed atlas.nii.gz --moved warped.nii.gz --model model.h5 --gpu 0
```

This will save the moved image to `warped.nii.gz`. To also save the predicted deformation field, use the `--save-warp` flag. Both npz or nifty files can be used as input/output in this script.


## Testing (measuring Dice scores)

To test the quality of a model by computing dice overlap between an atlas segmentation and warped test scan segmentations, run:

```
./scripts/tf/test.py --model model.h5 --atlas atlas.npz --scans scan01.npz scan02.npz scan03.npz --labels labels.npz
```

Just like for the training data, the atlas and test npz files include `vol` and `seg` parameters and the `labels.npz` file contains a list of corresponding anatomical labels to include in the computed dice score.

## Parameter choices

In the folder weights, you can find the weights of the 2 best semi-supervised models we developped. Their loss function is respectively MSE and NCC.
