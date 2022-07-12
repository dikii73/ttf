import sys
#import dlib
import torch
import cv2
#import pandas as pd
from motrackers import CentroidTracker, utils, IOUTracker
import numpy as np


# cfg
model_confidence = 0.4
iou_tresh = 0.5

# model 
model = torch.hub.load('ultralytics/yolov5' , 'custom', path='model/best.pt')
model.conf = model_confidence
model.eval()

#tracker
tracker = CentroidTracker()
tracker = IOUTracker(
    max_lost=25, 
    iou_threshold=iou_tresh, 
    min_detection_confidence=model_confidence, 
    max_detection_confidence=0.8,
    tracker_output_format='mot_challenge')

#video
video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)           

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    image = frame.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image).pandas().xyxy[0]
    if len(results) > 0:
        detection_bboxes = []
        detection_confidences = []
        detection_class_ids = []

        for row in results.iterrows():
            xmin, ymin, xmax, ymax, confidence, class_, name = row[1]
            box = [int(xmin), int(ymin), int(xmax), int(ymax)]
            print(xmin, ymin, xmax, ymax, confidence, class_, name)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255,0,0))
            cv2.putText(frame, name, (box[0], box[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
            
            box = utils.misc.xyxy2xywh(np.array(box))
            detection_bboxes.append(box)
            detection_confidences.append(confidence)
            detection_class_ids.append(class_)

    
        output_tracks = tracker.update(detection_bboxes, detection_confidences, detection_class_ids)
        frame = utils.draw_tracks(frame, output_tracks)
            
    cv2.imshow('', frame)
    cv2.waitKey(25)
