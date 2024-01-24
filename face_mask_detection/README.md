# Face Mask Detection system

This project is a Face Mask Detection System that utilizes computer vision techniques for detecting whether a person is wearing a face mask or not. The system is built using the OpenCV and TensorFlow libraries, and it incorporates a dataset from Kaggle to train the underlying model. The face detection process is achieved through the use of the haar_cascade.xml file, which is employed to identify faces in images. The extracted faces are then fed into the trained model to determine the presence or absence of a face mask.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)


## Installation

Install the python scripts provided in this repo. Download the dataset from [kaggle](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) and extract the images and annotations subfolders into the project folder. Also download the [Haar cascade classifier](https://github.com/enespolat25/OpenCV-1-2-3-4/blob/master/haarcascade-frontalface-default.xml) and keep it in the folder as well. Also create a new folder 'dataset' containing three empty subfolders 'with_mask', 'without_mask' and 'mask_weared_incorrect'.

## Usage

First run img_extraction.py file. The images will be extracted to their respective subfolders in the dataset folder. Next run the model_creation.py file to create the model. Finally run main.py to make inferences from a live video feed.

