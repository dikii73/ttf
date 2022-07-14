import sys
import torch
import cv2
import numpy as np
from tracker.tracker import Tracker as CentroidTracker
from tracker.utils import draw_tracks, xyxy2xywh

model_confidence = 0.6
iou_tresh = 0.3

# model 
model = torch.hub.load('ultralytics/yolov5' , 'custom', path='model/best.pt')
model.conf = model_confidence
model.eval()

#tracker
tracker = CentroidTracker(max_lost=25)


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
            bbox = np.array((int(xmin), int(ymin), int(xmax), int(ymax)))

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255))

            bbox = xyxy2xywh(bbox)
            detection_bboxes.append(bbox)
            detection_confidences.append(confidence)
            detection_class_ids.append(class_)

        output_tracks = tracker.update(detection_bboxes, detection_confidences, detection_class_ids)
        print(output_tracks)
        frame = draw_tracks(frame, output_tracks)
            
    cv2.imshow('', frame)
    cv2.waitKey(25)
