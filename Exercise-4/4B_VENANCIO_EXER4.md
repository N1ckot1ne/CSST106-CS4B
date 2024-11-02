# 4B_VENANCIO_EXER4

## Overview

This project implements object detection using various techniques including HOG-SVM, YOLO, and SSD models. The goal is to evaluate and compare the performance of these methods on a given dataset of images. The implementation is built using TensorFlow, OpenCV, and other essential libraries.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [License](#license)

## Requirements

To run this project, you'll need the following libraries:

- TensorFlow
- OpenCV
- scikit-learn
- Matplotlib

## Installation

You can install the required libraries using pip. Run the following command:

```bash
pip install tensorflow opencv-python scikit-learn matplotlib

Usage
Download the Models:

YOLO: Ensure you have the YOLO weights (yolov3.weights) and configuration file (yolov3.cfg). You can download these from the official YOLO website or GitHub repository.
SSD: Download the SSD MobileNet V2 model from the TensorFlow Model Zoo. Unzip the model and place the folder named ssd_mobilenet_v2_coco in the same directory as your script.
Update Image Path:

Update the image_paths variable in the script with the path to your image file.
Run the Code:

Execute the script in your Python environment. The performance of each detection method will be evaluated and displayed in a matplotlib window.
Models
This project uses the following object detection models:

HOG-SVM: A traditional object detection method using Histogram of Oriented Gradients (HOG) combined with a Support Vector Machine (SVM).
YOLO (You Only Look Once): A real-time object detection system that predicts bounding boxes and class probabilities for multiple objects in images.
SSD (Single Shot MultiBox Detector): A deep learning object detection model that uses a single deep neural network to predict bounding boxes and class scores.
License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments
TensorFlow Model Zoo for the SSD models.
OpenCV for image processing and computer vision functionalities.
The original authors of the YOLO and HOG-SVM algorithms for their contributions to the field of object detection.
