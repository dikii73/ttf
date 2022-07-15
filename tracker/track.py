import numpy as np

class Track:
    """
    Track containing attributes to track various objects.

    Args:
        frame_id (int): Camera frame id.
        track_id (int): Track Id
        bbox (numpy.ndarray): Bounding box pixel coordinates as (xmin, ymin, width, height) of the track.
        detection_confidence (float): Detection confidence of the object (probability).
        class_id (str or int): Class label id.
        lost (int): Number of times the object or track was not tracked by tracker in consecutive frames.
        iou_score (float): Intersection over union score.
        kwargs (dict): Additional key word arguments.

    """

    count = 0

    def __init__(
        self,
        track_id:int,
        frame_id:int,
        bbox:np.ndarray,
        detection_confidence:float,
        center:np.ndarray,
        class_id=None,
        lost=0,
        iou_score=0.,
        **kwargs
    ):
        Track.count += 1
        self.id = track_id

        self.detection_confidence_max = 0.
        self.lost = 0
        self.age = 0
        self.centers = []

        self.update(frame_id, bbox, detection_confidence, class_id=class_id, lost=lost, iou_score=iou_score, center=center, **kwargs)

        self.output = self.get_output

    def update(self, frame_id:int, bbox:np.ndarray, detection_confidence:float, center:np.ndarray, class_id=None, lost:int=0, iou_score:float=0., **kwargs) -> None:
        """
        Update the track.

        Args:
            frame_id (int): Camera frame id.
            bbox (numpy.ndarray): Bounding box pixel coordinates as (xmin, ymin, width, height) of the track.
            detection_confidence (float): Detection confidence of the object (probability).
            class_id (int or str): Class label id.
            lost (int): Number of times the object or track was not tracked by tracker in consecutive frames.
            iou_score (float): Intersection over union score.
            kwargs (dict): Additional key word arguments.
        """
        self.class_id = class_id
        self.bbox = np.array(bbox)
        self.detection_confidence = detection_confidence
        self.frame_id = frame_id
        self.iou_score = iou_score
        self.centers.append([*map(int, center)])

        if lost == 0:
            self.lost = 0
        else:
            self.lost += lost

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.detection_confidence_max = max(self.detection_confidence_max, detection_confidence)

        self.age += 1

    @property
    def centroid(self) -> np.array:
        """
        Return the centroid of the bounding box.

        Returns:
            numpy.ndarray: Centroid (x, y) of bounding box.

        """
        return np.array((self.bbox[0]+0.5*self.bbox[2], self.bbox[1]+0.5*self.bbox[3]))

    def get_output(self) -> tuple:
        """

        Returns:
            tuple: Tuple of 8 elements representing `(frame, id, bb_left, bb_top, bb_width, bb_height, conf, centers, age)`.

        """
        _tuple = (
            self.frame_id, self.id, self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], self.detection_confidence, self.centers, self.age
        )
        return _tuple