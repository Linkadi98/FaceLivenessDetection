<<<<<<< HEAD
# Liveness Detection

* This code trains and implements via video from the [pyimagesearch liveness detection blog](https://www.pyimagesearch.com/2019/03/11/liveness-detection-with-opencv)

* We modified the blog's original shallow CNN model to Resnet50 that can achieve better accuracy

<p align="center">
  <img src="https://github.com/joytsay/livenessDetection/blob/master/dataset/ezgif-1-085534fa4973.gif?raw=true" width="600">
</p>
<br>
<br>

## Get this code:
```
git clone https://github.com/joytsay/livenessDetection.git
cd livenessDetection
```

## Pre-Trained Model and Train/Test Videos:
Download from [here](https://drive.google.com/drive/folders/1Uj49JwLSAY4Q4v6UVMNF0u9hobGrJoWC?usp=sharing) and put in root folder (`/livenessDetection`)

## Setup Environment:
### Tested on Windows 10 [mini-conda](https://docs.conda.io/en/latest/miniconda.html) environment via

[Miniconda3-latest-Windows-x86_64.exe](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)

### Tested on MacOS M2 [mini-forge](https://github.com/conda-forge/miniforge) environment via
`brew install miniforge`

## Install Code:
### Option 1: Auto run bat files
run 01_XXX.bat files (01~05) sequentially:
```
01_install.bat
02_gather.bat
03_trainLiveness.bat
04_runLiveness.bat
05_webcam.bat
```
### Option 2: Cmd Line
```
conda create -n liveness python=3.8
conda activate liveness
pip install -r requirements.txt
```

```
# data pre-process
python3 gather_examples.py -i ./videos/fake.mp4 -o ./dataset/fake -d ./face_detector -c 0.9 -s 1 -f 0
python3 gather_examples.py -i ./videos/real.mp4 -o ./dataset/real -d ./face_detector -c 0.9 -s 1 -f 0
python3 train_liveness.py -d ./dataset -m liveness.model -l le.pickle

# python3 gather_examples.py -i ./videos/mask.mp4 -o ./dataset/mask -d ./face_detector -c 0.9 -s 1 -f 0

# run liveness model on test video
python3 liveness_demo.py -m liveness.model -l le.pickle -d ./face_detector -c 0.5
# press "q" to quit

# run liveness model on web cam
python3 webcam.py -m liveness.model -l le.pickle -d ./face_detector -c 0.5

```



## Reference
 The following link is the original pyimagesearch [liveness-detection-with-opencv example code](https://www.pyimagesearch.com/2019/03/11/liveness-detection-with-opencv)

### Find this project useful ? :heart:
* Support it by clicking the :star: button on the upper right of this page. :v:

### Credits
* The example has been taken from pyimagesearch liveness-detection-with-opencv example and modified model to Resnet50

### License
Copyright (C) 2020 Adrian Rosebrock, [PyImageSearch](https://www.pyimagesearch.com/2019/03/11/liveness-detection-with-opencv/), accessed on March 11, 2019

### Contributing to livenessDetection
Just make pull request. Thank you!
=======
# Face Liveness Detection using Depth Map Prediction

## About the Project

This is an application of a combination of Convolutional Neural Networks and Computer Vision to detect
between actual faces and fake faces in realtime environment. The image frame captured from webcam is passed over a pre-trained model. This model is trained on the depth map of images in the dataset. The depth map generation have been developed from a different CNN model.



## Requirements

* Python3
* Tensorflow
* dlib
* Keras
* numpy
* sklearn
* Imutils
* OpenCV 


## File Description

[main.py](https://github.com/anand498/Face-Liveness-Detection/blob/master/main.py):
This file is the main script that would call the predictperson function present in the utilr function

[training.py](https://github.com/anand498/Face-Liveness-Detection/blob/master/livenessdetect/training.py):
Along with the architecture script, this file includes various parameter tuning steps of the model.

[model.py](https://github.com/anand498/Face-Liveness-Detection/blob/master/livenessdetect/model.py) :
Has the main CNN architecture for training the dataset

## The Convolutional Neural Network

The network consists of **3** hidden conlvolutional layers with **relu** as the activation function. Finally it has **1** fully connected layer.

The network is trained with **10** epochs size with batch size **32**

The ratio of training to testing bifuracation is **75:25**


### How to use application in real time.


```
git clone https://github.com/anand498/Face-Liveness-Detection.git
pip install -r requirements.txt
python main.py
```
And you're good to go!

Don't forget to  :star:    the repo if I made your life easier with this. :wink:



>>>>>>> 0baeb957d0d2b32de6d63456f8fb330c0e22c08e
