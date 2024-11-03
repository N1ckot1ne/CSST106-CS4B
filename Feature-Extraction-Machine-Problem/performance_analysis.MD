# **Harris Corner Detection**
This program demonstrates the implementation of the Harris Corner Detection algorithm in Python using OpenCV. The program identifies and highlights corners in an input image, marking them in red for easy visualization.

Requirements
Ensure the following libraries are installed before running the script:

* OpenCV
* NumPy
* Matplotlib

You can install these dependencies using pip:
```python
pip install opencv-python-headless numpy matplotlib
```
**Usage**

* Prepare Your Image: Save the image you want to analyze in the same directory as the script, or update the file path in the harris_corner_detection function.

* Run the Script: Execute the script to display the original image alongside the processed image, which will have detected corners marked in red.

* Example: 

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def harris_corner_detection(image_path="4B_VENANCIO_MATCHING.jpg"):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_copy = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if img is None or img_copy is None:
        print(f"Error: Could not load image at path '{image_path}'. Please check the file path.")
        return

    # Convert to float and apply Harris Corner Detection
    gray = np.float32(img)
    harris_corners = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Mark the corners in red
    img_copy[harris_corners > 0.01 * harris_corners.max()] = [0, 0, 255]

    # Display images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.axis('off')
    plt.subplot(1, 2, 2), plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.title('Harris Corner Detection'), plt.axis('off')
    plt.show()

# Run the function
harris_corner_detection()
```

# **HOG Feature Extraction**
This Python script demonstrates how to perform Histogram of Oriented Gradients (HOG) feature extraction on an image using OpenCV and the skimage.feature library. HOG is a method commonly used for object detection and computer vision tasks, as it captures the structure or shape of an object by encoding gradients or changes in intensity.
**Requirements**
Install the required libraries before running the script:
* OpenCV
* scikit-image
* Matplotlib

Install using pip:

```python
pip install opencv-python-headless scikit-image matplotlib
```
**Usage**
*Prepare Your Image: Save the image you want to analyze in the same directory as the script, or update the file path in the hog_feature_extraction function.
*Run the Script: Execute the script to display the original image alongside the image with visualized HOG features.
*Example: 

```python
from skimage.feature import hog
import cv2
import matplotlib.pyplot as plt

def hog_feature_extraction(image_path="4B_VENANCIO_MATCHING.jpg"):
    img = cv2.imread(image_path)
    
    # Check if the image loaded successfully
    if img is None:
        print(f"Error: Could not load image at path '{image_path}'. Please check the file path.")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Extract HOG features and HOG visualization
    hog_features, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), visualize=True)

    # Display original and HOG images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image'), plt.axis('off')
    plt.subplot(1, 2, 2), plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Features'), plt.axis('off')
    plt.show()

# Run the function
hog_feature_extraction()
'''

# **ORB Feature Matching**
This Python script demonstrates how to perform ORB (Oriented FAST and Rotated BRIEF) feature extraction and feature matching between two images using the FLANN-based matcher. ORB is a computationally efficient alternative to SIFT and SURF and is widely used in real-time applications due to its speed and robustness.

**Requirements**

* OpenCV
* Matplotlib

Installation: 
```python
pip install opencv-python-headless matplotlib
```

**Usage**
*Prepare Your Images: Ensure you have two images in your directory. Update the image paths in the orb_feature_matching function if they are named differently.
*Run the Script: Execute the script to display the original images with matched keypoints.
*Example:

```python
import cv2
import matplotlib.pyplot as plt

def orb_feature_matching(image_path1="4B_VENANCIO_MATCHING.jpg", image_path2="sample.jpg"):
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print(f"Error: Could not load one or both images. Please check the file paths.")
        return
    
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=2)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)

    plt.figure(figsize=(10, 5))
    plt.imshow(img_matches)
    plt.title('ORB Feature Matching'), plt.axis('off')
    plt.show()

# Run the function
orb_feature_matching()
'''

