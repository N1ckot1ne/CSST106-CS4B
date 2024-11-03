# **COCO Dataset Download and Extraction**
This guide explains how to download and extract the COCO 2017 training dataset for use in machine learning or computer vision projects.

**Steps to Download and Extract COCO 2017 Train Set**

**1.  Create Directory.** create a directory structure to store the dataset:

```bash
mkdir -p datasets/coco
```

**2.  Download the Dataset.** Use curl to download the COCO 2017 training data (train2017.zip) into the datasets/coco directory:

```bash
curl -L "http://images.cocodataset.org/zips/train2017.zip" -o datasets/coco/train2017.zip
```

**3.  Extract the Dataset. ** Once downloaded, unzip the file into the datasets/coco directory:

```bash
unzip -q datasets/coco/train2017.zip -d datasets/coco/
```

This will extract all images in the train2017 set to datasets/coco/train2017/.

**Directory Structure.**After completing these steps, your directory structure should look like this:

```python
datasets/
└── coco/
    ├── train2017.zip       # Original zip file (optional to keep)
    └── train2017/          # Extracted images
```
# **Implementing HOG-SVM**
This project demonstrates how to use the HOG (Histogram of Oriented Gradients) descriptor and a pre-trained SVM detector in OpenCV to detect people in an image.

**Requirements**
* Python (>=3.6 recommended)
* OpenCV (cv2)

**Setup**
1. Install OpenCV. 

```python
pip install opencv-python
```
2. Prepare an Input Image. Place an example image at datasets/coco/processed/example_image.jpg or update the image_path variable in the code with the path to your desired image.

**How it Works**
The code initializes a HOG descriptor, sets up a pre-trained SVM detector for human detection, and then processes the specified image to detect people. Detected people are highlighted with bounding boxes.

Code Explanation
```python
import cv2

# Initialize HOG descriptor and set SVM detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load an example image and perform detection
image_path = "datasets/coco/processed/example_image.jpg"
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Unable to load image at {image_path}")
else:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect people in the image
    boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8))

    # Draw bounding boxes around detected people
    for (x, y, w, h) in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the output image
    cv2.imwrite("output.jpg", image)
```

**Key Points**
* **HOG Descriptor**: Initializes the HOG descriptor and sets up the default people detector.
* **Detecting People**: The detectMultiScale function finds people in the image.
* **Drawing Bounding Boxes**: For each detection, a bounding box is drawn around the detected individual.

*Notes*
* **Error Handling**: If the image fails to load, an error message is displayed.
* **Output**: The processed image with bounding boxes is saved as output.jpg

**Running the Code**
1. Ensure your image path is correctly set in image_path.
2. Run the code to detect the image
3. Check output.jpg for the detection results.

**Additional Information**
The HOG + SVM combination is popular for real-time people detection but may have limitations in complex scenes. This setup provides a starting point for experimenting with HOG-based detection.

# **Displaying Output Image using Matplotlib**
1. If you haven't installed OpenCV and Matplotlib, you can install them using pip:

```python
pip install opencv-python matplotlib
```

2. Load and Display the Image. This code snippet reads an image (saved as output.jpg), converts it to RGB, and displays it without axes using Matplotlib:
```python
from matplotlib import pyplot as plt
import cv2

# Load and display the output image
image_output = cv2.imread("output.jpg")
plt.imshow(cv2.cvtColor(image_output, cv2.COLOR_BGR2RGB))
plt.axis("off")  # Hide axes
plt.show()
```
**Explanation**
* **Image Conversion**: OpenCV loads images in BGR format, so we convert it to RGB for accurate display with Matplotlib.
* **Display Settings**: plt.axis("off") hides the axes for a cleaner view of the image.

# **Image Classification using TensorFlow and MobileNetV2**
This guide demonstrates how to use a pre-trained MobileNetV2 model for image classification in TensorFlow. Although MobileNetV2 is not a YOLO model, it serves as a placeholder for quick object classification.

**Requirements**
* TensorFlow

*Setup*
1. Install TensorFlow
```python
pip install tensorflow
```
