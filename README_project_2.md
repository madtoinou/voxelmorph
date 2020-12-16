
# Machine learning (CS-433): ML4 science

_____________________________________________________________________

## Project 2: Deformable image registration of worm brains using Voxelmorph method 

_Adapted for Professor Sahand of LPBS Lab at EPFL by Ines, Loic and Antoine


### Datasets

In general, the datasets contain frames of worm brain where xxx neurons of interest were stained with red fluorescent proteins which are visible in the _red channel_. In addition, a genetically encoded calcium indicator (GCaMP) realeasing green fluorescence upon activation was used to detect neuronal activity and is visible in the _green channel_. The data were acquired by confocal microscopy using UP & DOWN scanning. All the data were stored in `.h5` files, a widely used format for biological data.

In this project, 2 datasets were provided depending of the required tasks and can be described more explicitly as follow:

##### Dataset of pre-aligned data:

The pre-aligned data consists of 1715 frames, 118 of which are labelled.  The  red  channel  only  is  available  and  each frame  has  a  volume  of  112x112x32. These data were already preprocessed using the algorithm of the LPBS lab, consisting mainly in a binary thresholding followed by affine registration using the _Jian & Vermuri_ algorithm. Finally cropping was performed to obtain the 112x112x32 from 512x512x35 frames. 

##### Dataset of raw data:

The raw data consists of 570 frames of size 512x512x35, with no labelled data. Both red and green channels data are available and no preprocessing was performed on data before we obtained them.

### Handling of `.h5` files

We provide here a small guide on how to handle `.h5` files provided in this work, to allow the reader to access data easily:

1. The command `hf = h5py.File("../data")` allows to recover the `.h5` file and store it in the variable `hf`. `"../data"` corresponds to the path to your data.
2. The command `hf.keys()` allows to access the keys of the file, where datasets containing frames. 
3. The commands `hf.get(key)["frame"][0]` or `hf.get(key)["frame"][1]` allows to access a specific frame of the _red channel_ (corresponding to 0) and _green channel_ (corresponding to 1) encoded in a specific key. 
   Accordingly, `hf.get(key)["mask"][0]` or `hf.get(key)["mask"][1]` allows to access a specific mask (labelled data) of the _red channel_ (corresponding to 0) and _green channel_ (corresponding to 1) encoded in a specific key.

More information on handling of `.h5` files can be found on the [HDF5 for python user manual](https://docs.h5py.org/en/stable/).

## Tutorial

Visit the [VoxelMorph github repository](https://github.com/voxelmorph/voxelmorph) and [VoxelMorph tutorial](http://tutorial.voxelmorph.net/) to learn about VoxelMorph and Learning-based Registration


## Instructions

To run our code, install the requirements listed in `setup.py`.

### Useful files

The code is separated in 2 distinctive files containing all the functions to reproduce our results:

>1. utilitary.py
>2. create_npz.py
>3. 3d_training_unsup.py
>4. 3d_training_semipsup.py

-------------------------------------------------------------TO REMOVE--------------------------------------------------------------------- 
#### utilitary.py:

This file contains all the functions required to reproduce our preprocessing pipeline.

- "IMPLEMENTATIONS" is composed of the 6 functions `least_squares_GD`, `least_squares_SGD`, `least_squares`, `ridge_regression`, `logistic regression` and `reg_logistic_regression` constituing a toolbox for development of the regression model.

- "UTILITARIES" contains complementary functions to ensure good working of the methods present in "IMPLEMENTATION" section, as well as functions needed for prediction and loading of datasets.

- "PREPROCESSING" contains all the preprocessing steps used in this work to optimize the model's performance.


#### run.py:

`run.py` allows to reproduce the best prediction accuracy stated in the report. The optimal hyperparameters are already provided. Function `load_csv_data()` provided by the teachers for loading train set, predict labels and create a submission file in `.csv` format is also given in this file.

-------------------------------------------------------------TO REMOVE---------------------------------------------------------------------

### Model's basic operations

The model implemented in `run.py` loads the training set provided in the DATA_TRAIN_PATH (see Code execution) and several preprocessing steps are performed on this dataset in this order: 
1. Splitting of the features based on the PRI_jet_num categories (0,1 or 2&3) 
2. logarithmic transformation of selected features 
3. Polynomial augmentation of the features 
4. Standardization of the features. 
    
Afterwards, the model is trained using the ridge regression algorithm and weights are obtained. These weights are used to predict each labels of the splitted dataset and the predictions are finally merged and the submission file is created.


# Code execution

## 1) Data transformation before Training

1) Transform the data into npz. __Note that the call to `python` may varie depending on the version of python you are using.__

```
python create_npz.py
```

## 2) Training of data (2 possibilities)

###  2.1) Unsupervised training

__Optimized in this work:__

For '--image-loss' : 'mse' = mean-squared error or 'ncc' = normalized cross-correlation
'--grad-loss-weight' : the L2-regularizer
'--savename' : store the results in a `.txt` file with name `name`
'--epochs' :  define the number of epochs. Default is 32 (__best?__)
'--steps-per-epoch' : define the number of steps in each epochs (corresponds to final epoch). Default is 250 (__best?__)
'--batch-size' : define the batch size. Default is 4 (mini-batch gradient descent)
'--atlas' : define the reference (fixed) frame default = '904', help='optional atlas to perform scan-to-atlas training') (__oui finalement?__)

__Let by default (not optimized):__

'--gpu', default='0'
'--initial-epoch' : define from which epoch to start
'--lr' : learning rate for Adam optimization algorithm. Default is 1e-4
'--seed' : seed defined to perform the results in the project. Default is 336699
'--enc' : encoder of the Unet. Default is 16 32 32 32
'--dec' : decoder of the Unet. Default: 32 32 32 32 32 16 16
'--int-steps' : number of integration steps. Default is 7
'--int-downsize' : flow downsample factor for integration. Default is 2
'--dice-loss' : the auxiliary loss function using dice score as metric

2) Command to enter in the terminal for reproducible results:

```
python 3d_training_unsup.py --image-loss 'mse' --grad-loss-weight 0.05 --savename 'name' --epochs 32 --steps-per-epoch 250
```

### 2.2) Semisupervised training

__Optimized in this work:__

For '--image-loss' : 'mse' = mean-squared error or 'ncc' = normalized cross-correlation
'--grad-loss-weight' : the L2-regularizer (`lambda`)
'--savename' : store the results in a `.txt` file with name `name`
'--epochs' :  define the number of epochs. Default is 32 (__best?__)
'--steps-per-epoch' : define the number of steps in each epochs (corresponds to final epoch). Default is 250 (__best?__)
'--batch-size' : define the batch size. Default is 4 (mini-batch gradient descent)
'--atlas' : define the reference (fixed) frame default = '904', help='optional atlas to perform scan-to-atlas training')

__Let by default (not optimized):__

'--gpu', default='0'
'--initial-epoch' : define from which epoch to start
'--lr' : learning rate for Adam optimization algorithm. Default is 1e-4
'--seed' : seed defined to perform the results in the project. Default is 336699
'--enc' : encoder of the Unet. Default is 16 32 32 32
'--dec' : decoder of the Unet. Default: 32 32 32 32 32 16 16
'--int-steps' : number of integration steps. Default is 7
'--int-downsize' : flow downsample factor for integration. Default is 2
'--dice-loss' : the auxiliary loss function using dice score as metric

2) Command to enter in the terminal for reproducible results:

```
python 3d_training_semipsup.py --image-loss 'mse' --grad-loss-weight 0.05 --savename 'name' --epochs 32 --steps-per-epoch 250
```
## 3) Testing of data using Dice score metric

