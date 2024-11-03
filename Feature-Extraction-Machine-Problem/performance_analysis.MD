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

Prepare Your Image: Save the image you want to analyze in the same directory as the script, or update the file path in the harris_corner_detection function.
Run the Script: Execute the script to display the original image alongside the processed image, which will have detected corners marked in red.
Example: 

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
Prepare Your Image: Save the image you want to analyze in the same directory as the script, or update the file path in the hog_feature_extraction function.
Run the Script: Execute the script to display the original image alongside the image with visualized HOG features.
Example: 

``` python
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
```

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

Prepare Your Images: Ensure you have two images in your directory. Update the image paths in the orb_feature_matching function if they are named differently.
Run the Script: Execute the script to display the original images with matched keypoints.
Example:

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
```
# **SIFT Feature Extraction**
This Python script demonstrates how to perform feature extraction using the SIFT (Scale-Invariant Feature Transform) algorithm. SIFT is a powerful technique for detecting and describing local features in images, making it widely used in various computer vision applications, including object recognition and image stitching.

**Requirements**
Before running the script, ensure that you have the following libraries:
* OpenCV
* Matplotlib

Instalation:
```python
pip install opencv-python-headless matplotlib
```
**Usage**
Prepare Your Images: Ensure you have two images in your directory. Update the image paths in the sift_feature_extraction function if they are named differently.
Run the Script: Execute the script to display the first image with SIFT keypoints highlighted.
Example:

```python
import cv2
import matplotlib.pyplot as plt

def sift_feature_extraction(image_path1="4B_VENANCIO_MATCHING.jpg", image_path2="sample.jpg"):
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Error: Could not load one or both images. Please check the file paths.")
        return

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors with SIFT
    kp_sift1, des_sift1 = sift.detectAndCompute(img1, None)
    kp_sift2, des_sift2 = sift.detectAndCompute(img2, None)

    # Draw SIFT keypoints on the first image
    img_sift1 = cv2.drawKeypoints(img1, kp_sift1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display SIFT keypoints for the first image
    plt.figure(figsize=(10, 5))
    plt.imshow(img_sift1, cmap='gray')
    plt.title('SIFT Keypoints')
    plt.axis('off')
    plt.show()

# Run the function
sift_feature_extraction()
```
# **Image Segmentation using Watershed Algorithm**
This Python script demonstrates how to perform image segmentation using the Watershed algorithm with OpenCV. The script processes an input image, identifies distinct regions, and visualizes the segmentation results by marking the boundaries in red.

Requirements
Ensure the following libraries are installed before running the script:

* OpenCV
* NumPy
* Matplotlib

Usage
Prepare Your Image: Save the image you want to analyze in the same directory as the script, or update the file path in the watershed_segmentation function.
Run the Script: Execute the script to display the original image with the segmented regions highlighted.

**Code**
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def watershed_segmentation(image_path="4B_VENANCIO_MATCHING.jpg"):
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image at path '{image_path}'. Please check the file path.")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Watershed Segmentation'), plt.axis('off')
    plt.show()

# Run the function
watershed_segmentation()
```
