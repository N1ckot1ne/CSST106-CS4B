# Image Processing Exercises with OpenCV and Python

This repository contains a collection of image processing exercises using OpenCV and Python. Each exercise demonstrates different computer vision techniques commonly used in object detection, feature extraction, segmentation, and matching.

---

## Table of Contents

1. [Exercise 1: Harris Corner Detection](#exercise-1-harris-corner-detection)
2. [Exercise 2: HOG Feature Extraction](#exercise-2-hog-feature-extraction)
3. [Exercise 3: FAST Keypoint Detection](#exercise-3-fast-keypoint-detection)
4. [Exercise 4: Feature Matching with ORB and FLANN](#exercise-4-feature-matching-with-orb-and-flann)
5. [Exercise 5: Image Segmentation using Watershed Algorithm](#exercise-5-image-segmentation-using-watershed-algorithm)

---

## Exercise 1: Harris Corner Detection

**Description**: Harris Corner Detection is a classic algorithm used to detect corners or points of interest in an image, particularly where object edges intersect.

**Key Steps**:
1. Load an image and convert it to grayscale.
2. Apply the Harris Corner Detection method to identify corners.
3. Visualize the corners by marking them on the original image.

**Code**: [Harris Corner Detection Code](exercise_1_harris.py)

---

## Exercise 2: HOG Feature Extraction

**Description**: The HOG (Histogram of Oriented Gradients) descriptor is widely used for object detection, especially in applications like human detection. It focuses on the gradient structure of objects, capturing edges and shapes.

**Key Steps**:
1. Load an image and convert it to grayscale.
2. Apply the HOG descriptor to extract features.
3. Visualize the gradient orientations to understand the structure of the detected object.

**Code**: [HOG Feature Extraction Code](exercise_2_hog.py)

---

## Exercise 3: FAST Keypoint Detection

**Description**: FAST (Features from Accelerated Segment Test) is a quick and computationally efficient keypoint detector, ideal for real-time applications like robotics and mobile vision.

**Key Steps**:
1. Load an image and convert it to grayscale.
2. Use the FAST algorithm to detect keypoints.
3. Visualize the detected keypoints by marking them on the image.

**Code**: [FAST Keypoint Detection Code](exercise_3_fast.py)

---

## Exercise 4: Feature Matching with ORB and FLANN

**Description**: This exercise uses ORB (Oriented FAST and Rotated BRIEF) descriptors to find and match features between two images. The FLANN (Fast Library for Approximate Nearest Neighbors) matcher speeds up the process, which is especially useful for large datasets.

**Key Steps**:
1. Load two images and convert them to grayscale.
2. Extract keypoints and descriptors using ORB.
3. Match features between the two images using the FLANN matcher.
4. Display the matched features to observe common points between the two images.

**Code**: [ORB and FLANN Feature Matching Code](exercise_4_orb_flann.py)

---

## Exercise 5: Image Segmentation using Watershed Algorithm

**Description**: The Watershed algorithm segments an image into distinct regions, which is useful for separating overlapping objects.

**Key Steps**:
1. Load an image and convert it to grayscale.
2. Apply a binary threshold to convert the image into a binary format.
3. Use morphological operations to remove noise and identify background and foreground areas.
4. Apply the Watershed algorithm to segment the image into regions.
5. Visualize the segmented regions by marking boundaries between different objects.

**Code**: [Watershed Image Segmentation Code](exercise_5_watershed.py)

---

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- Scikit-image (for HOG extraction)

Install the necessary libraries using:
```bash
pip install opencv-python numpy matplotlib scikit-image
