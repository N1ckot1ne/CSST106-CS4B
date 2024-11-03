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
