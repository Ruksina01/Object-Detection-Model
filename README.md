# Object Detection and Counting Using Deep Learning 🤖🔍

### Overview

This project implements an object detection and counting system using deep learning techniques. The model detects and counts objects in images or video streams, providing valuable insights for various applications. 📊✨

### Table of Contents

1. [Prerequisites](#prerequisites)
2. [Dataset Preparation](#dataset-preparation)
3. [Data Preprocessing](#data-preprocessing)
4. [YAML File Creation](#yaml-file-creation)
5. [Training the Model](#training-the-model)
6. [Making Predictions](#making-predictions)
7. [Results](#results)

### 1. Prerequisites 🛠️

Ensure you have the following software and libraries installed:

- Python 3.x
- TensorFlow or PyTorch (depending on the framework you choose)
- OpenCV
- NumPy
- YOLOv5 (or your chosen model)
- Additional dependencies (as specified in the `requirements.txt` file)

# Dataset Preparation Guide

## Dataset Overview
- **Source**: [Kaggle: PESMOD Data](https://www.kaggle.com/datasets/ruksinakhan/pesmod-data)
- **Description**: The dataset contains data related to PESMOD, which can be used for various analytical and modeling purposes.

## Steps for Dataset Preparation

### 1. Downloading the Dataset
- Go to the provided Kaggle link.
- Sign in to your Kaggle account (or create one if you don’t have it).
- Download the dataset files to your local machine.

### 2. Exploring the Dataset
- Load the dataset using a data analysis library (e.g., Pandas in Python).
- Inspect the first few rows to understand its structure:
  ```  python
  import pandas as pd

  # Load the dataset
  data = pd.read_csv('path/to/your/dataset.csv')
  # Display the first few rows
  print(data.head())
  ```

### 2. Dataset Preparation 📁

1. **Dataset Structure**:
   - Organize your dataset in the following structure:

2. **Data Annotation**:
   - Each label file should contain the object classes in YOLO format (class_id x_center y_center width height).
     

### 3. Data Preprocessing 🔄

1. **Load and Resize Images**:
   - Load images and resize them to the input size required by your model (e.g., 640x640 for YOLO).

2. **Normalization**:
   - Normalize the pixel values (scale them between 0 and 1) to improve training performance.

3. **Data Augmentation (Optional)**:
   - Apply data augmentation techniques such as flipping, rotation, or color jitter to increase dataset diversity.

4. **Split Dataset**:
   - **Purpose**: To prepare the dataset for training machine learning models by dividing it into three distinct sets:
   - **Training Set**: Used to train the model (typically 70% of the dataset).
   - **Validation Set**: Used to tune the model’s hyperparameters (typically 15% of the dataset).
   - **Testing Set**: Used to evaluate the model's performance (typically 15% of the dataset).
     
   - Execute the script to see how it performs.
   - You can run the file using the command line:
     ```bash
     savingsplitdataset.ipynb
     ```
     
### 4. YAML File Creation 📄

Create a YAML file (e.g., `dataset.yaml`) to define your dataset paths and classes. Here’s an example structure:

```yaml
path: /kaggle/input/pesmodsplit/split_dataset
train: /kaggle/input/pesmodsplit/split_dataset/train/images
val: /kaggle/input/pesmodsplit/split_dataset/val/images
test: /kaggle/input/pesmodsplit/split_dataset/test/images

names:
  0: 'Pexels-Elliot-road'
  1: 'Pexels-Grisha-snow'
  2: 'Pexels-Marian'
  3: 'Pexels-Miksanskiy'
  4: 'Pexels-Shuraev-trekking'
  5: 'Pexels-Welton'
  6: 'Pexels-Wolfgang'
  7: 'Pexels-Zaborski'
```

### 5. Training the Model 🏋️‍♀️

1. **Load the Model:**
   - Choose a pre-trained model (e.g., YOLOv8, Yolov11, or any) and load it
     ```
     git clone https://github.com/ultralytics/yolov5
     cd yolov5
     ```
2. **Install Requirements:**
   - Install the necessary dependencies
     ```
     pip install -r requirements.txt
     ```
3. **Start Training:**
   - Run the training script to start the training process. Monitor the training loss and validation metrics.
     ```
     python train.py --img 640 --batch 16 --epochs 50 --data dataset.yaml --weights yolov5s.pt
     ```

### 6. Making Predictions 🎯

1. **Load the Model:**
   - Load the trained weights of your model for inference.
2. **Prepare Input:**
  - Load an image or video stream and preprocess it similar to training.
3. **Run Inference:**
  - Use the model to predict objects in the input data.
    ```
    python detect.py --weights path/to/trained_weights.pt --img 640 --conf 0.25 --source path/to/image_or_video
    ```
    
### 7. Results 📈
After running predictions, visualize the results by drawing bounding boxes and displaying counts of detected objects.

1. **Visualize Predictions:**
  - Draw bounding boxes and class labels on the images or video frames.
2. **Output Counts:**
  - Count the number of detected objects and display the results


