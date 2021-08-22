import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time

# Load pytorch model -ultralytics Yolov5s (online approach)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# Load model - offline approach
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False)
# model.load_state_dict(torch.load('yolov5s.pt')['state_dict'])

# To load
# model = torch.jit.load('yolov5s.pt').eval().to(device)

# Load from img
# img = cv2.imread('cars.jpg')
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
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()