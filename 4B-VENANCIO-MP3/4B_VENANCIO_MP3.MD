# **Feature Matching and Image Alignment Using SIFT, AKAZE, and ORB in OpenCV**

This project demonstrates the use of SIFT, AKAZE, and ORB feature detection and matching algorithms in OpenCV. The main goals are to compare the performance of these algorithms in terms of accuracy and speed and to use the matched points to align two images based on homography transformation.

**Table of Contents**

* Requirements
* Project Structure
* Usage
* Feature Detection and Matching Algorithms
*Conclusion

**Requirements**
* Python 3.x
* OpenCV (with contrib modules for SIFT)
* NumPy
* Matplotlib

**Project Structure**

* sample1.jpg and sample2.jpg: Input images to perform feature matching and alignment on.
* feature_matching.py: Main script with implementations for SIFT, AKAZE, and ORB feature extraction, descriptor computation, and feature * matching. It includes visualization of keypoints and matches and image alignment.
