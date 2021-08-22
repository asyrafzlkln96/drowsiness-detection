# Custom object detection
# Load custom model that is trained 

import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time
import os

# Load custom trained model for drowsiness detection
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp3/weights/last.pt', force_reload=True)

# # Load from image
# img = os.path.join('data', 'images', 'drowsy.96a7c274-033d-11ec-9d90-7085c282a031.jpg')
# # results = model(img)
# # results.print()
# # # img = cv2.imread('cars.jpg')
# results = model(img)
    
# cv2.imshow('YOLO', np.squeeze(results.render()))
# cv2.waitKey(0)

# Read from webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Make detections 
    results = model(frame)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    print('Result of detection is: ',results.pandas().xyxy[0].name)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()