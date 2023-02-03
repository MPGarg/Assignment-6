# Assignment-6

## Problem Statement

![image](https://user-images.githubusercontent.com/120099863/216579996-6a432d27-6a8c-4417-9a83-9a331abe4f04.png)

## Code Details

Code is devided into five files. Four of them are py files & one ipynb file. Link for them are as below:
  * model_cifar10.py [notebook](model_cifar10.py)
  * model_transpose.py [notebook](model_transpose.py)
  * train_test.py [notebook](train_test.py)
  * utility.py [notebook](utility.py)
  * EVA8_Assigment_6.ipynb [notebook](EVA8_Assigment_6.ipynb)
  
All of these py files are imported into EVA8_Assigment_6.ipynb after uploading in google drive. model_cifar10.py contains a class by name Net that creates nueral network.

Following is the list of functions that are called for different purpose:
  * load_data: Load transformations for mean and standard deviation calculation
  * set_albumen_params: Albumentation library transformation
  * tl_ts_mod: Apply augmentation to dataset
  * train_test_model: Train & test function
  * display_incorrect_pred: Show incorrect predictions

## Model Design

Layers for Model:
 * Convolution Block 1
 * Transition Block 1
 * Convolution Block 2
 * Transition Block 2
 * Convolution Block 3 (It contains Depthwise Separable Convolution & Dilated Convolution)
 * Transition Block 3
 * Convolution Block 4
 * GAP

Summary of Model:

![image](https://user-images.githubusercontent.com/120099863/216589699-753def6b-c3af-4110-a830-049cb3bbbcca.png)
![image](https://user-images.githubusercontent.com/120099863/216589870-5a636741-da01-40ad-a02f-f67fa832172f.png)


Total Trainable Parameters: 79,684
Best Training Accuracy: 79.16%
Best Test Accuracy: 85.00% (Epoch: 192)

Training Log (last few steps):

![image](https://user-images.githubusercontent.com/120099863/216589237-7c887547-83e8-4850-b53f-7598ae29f9a1.png)




