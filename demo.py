import sys
import torch
import cv2
import numpy as np
from tracker.iou_tracker import IOUTracker
from tracker.utils import draw_tracks, xyxy2xywh

class Detect():
    def __init__(self, video:str):
        self.model_confidence = 0.6
        self.model_iou = 0.45
        self.model = torch.hub.load('ultralytics/yolov5' , 'custom', path='model/best.pt')
        self.model.conf = self.model_confidence
        self.model.iou = self.model_iou

        self.iou_tresh = 0.3
        self.max_lost = 10
        self.min_detection_confidence = self.model_confidence
        self.max_detection_confidence = 0.95
        self.tracker = IOUTracker(
            max_lost=self.max_lost, 
            iou_threshold=self.iou_tresh, 
            min_detection_confidence=self.min_detection_confidence, 
            max_detection_confidence=self.max_detection_confidence)
        
        self.video = video
        self.cap = cv2.VideoCapture(self.video)

    def run(self) -> None:
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
    
            image = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)

            results = self.model(image).pandas().xyxy[0]
            if len(results) > 0:
                detection_bboxes = []
                detection_confidences = []
                detection_class_ids = []

                for row in results.iterrows():
                    xmin, ymin, xmax, ymax, confidence, class_, _ = row[1]
                    bbox = np.array((int(xmin), int(ymin), int(xmax), int(ymax)))

                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,0,255))

                    bbox = xyxy2xywh(bbox)
                    detection_bboxes.append(bbox)
                    detection_confidences.append(confidence)
                    detection_class_ids.append(class_)

                output_tracks = self.tracker.update(detection_bboxes, detection_confidences, detection_class_ids)
                frame = draw_tracks(frame, output_tracks)
                    
            cv2.imshow('', frame)
            cv2.waitKey(25)


if __name__ == '__main__':
    video = sys.argv[1]
    det = Detect(video=video)
    det.run()