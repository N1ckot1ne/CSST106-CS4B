# **Image Processing Techniques**
This project demonstrates various fundamental image processing techniques using OpenCV in Google Colab, including image transformations (scaling and rotation) and filtering techniques (blurring and edge detection). These methods provide foundational skills for manipulating and analyzing images, applicable to many fields like computer vision, AI, and digital media.

**CONTENTS**

* Image Transformations: Scaling and rotating images.
* Filtering Techniques: Applying Gaussian blurring and edge detection.
* Scenario-Based Problem-Solving: Combining techniques to solve practical tasks.

**Installation**

```python
!pip install opencv-python-headless
```
# **1. Upload an Image**

```python
from google.colab import files
uploaded = files.upload()  # Allows image upload

# Load the uploaded image
image = cv2.imread('sample.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
```

# **2. Image Transformations**

**Scaling**
```python
scale_percent = 50  # Scale to 50%
dim = (int(image.shape[1] * scale_percent / 100), int(image.shape[0] * scale_percent / 100))
resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
```

**Rotation**
```python
angle = 45
center = (image.shape[1] // 2, image.shape[0] // 2)
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
```

# **3. Filtering Techniques**

**Blurring**
```python
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
```

**Edge Detection**
```python
edges = cv2.Canny(image, 100, 200)
```

# **4. Scenario-Based Problem-Solving**
```python
# Resize
scale_percent = 75
resized_image = cv2.resize(image, (int(image.shape[1] * scale_percent / 100), int(image.shape[0] * scale_percent / 100)), interpolation=cv2.INTER_AREA)

# Apply Gaussian blur
blurred_resized_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

# Detect edges
edges_resized_blurred = cv2.Canny(blurred_resized_image, 100, 200)
```

**Display Results**
```python
import matplotlib.pyplot as plt

# Example to display an image
plt.imshow(resized_image)
plt.axis('off')
plt.show()
```
