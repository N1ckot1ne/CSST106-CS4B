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
import cv2
import numpy as np
import matplotlib.pyplot as plt

```python
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
