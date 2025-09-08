# Skincare Product Detection using RCNN

## Introduction
Skincare product detection is an object detection task to identify products such as creams, serums, toners, or masks from images.  
Using **RCNN (Region-based Convolutional Neural Networks)**, we can detect product types and localize them with bounding boxes.

## How It Works
1. Input an image containing skincare products.  
2. Generate region proposals (possible object locations).  
3. Extract features using CNN.  
4. Classify the region as a specific product type and refine bounding boxes.  

## Python Implementation (Simplified Example)
Below is a simple example using **TensorFlow Object Detection API** with Faster R-CNN pre-trained weights.  
This can be fine-tuned on a custom skincare product dataset.

```python
import tensorflow as tf
import cv2
import numpy as np

# Load pre-trained Faster R-CNN model (from TensorFlow Model Zoo)
model = tf.saved_model.load("ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/saved_model")

# Load an image containing skincare products
image = cv2.imread("skincare.jpg")
input_tensor = tf.convert_to_tensor([image], dtype=tf.uint8)

# Perform detection
detections = model(input_tensor)

# Extract results
boxes = detections["detection_boxes"][0].numpy()
scores = detections["detection_scores"][0].numpy()
classes = detections["detection_classes"][0].numpy().astype(int)

# Draw detected skincare products
for i in range(len(scores)):
    if scores[i] > 0.5:  # confidence threshold
        y1, x1, y2, x2 = boxes[i]
        h, w, _ = image.shape
        cv2.rectangle(image, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (255,0,0), 2)
        cv2.putText(image, f"Product {classes[i]}", (int(x1*w), int(y1*h)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

cv2.imshow("Skincare Detection", image)
cv2.waitKey(0)
