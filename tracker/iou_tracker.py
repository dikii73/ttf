from tracker.utils import iou_xywh as iou, get_centroid
from tracker.tracker import Tracker

class IOUTracker(Tracker):
    """
    Intersection over Union Tracker.

    Args:
        max_lost (int): Maximum number of consecutive frames object was not detected.
        tracker_output_format (str): Output format of the tracker.
        iou_threshold (float): Intersection over union minimum value.
    """

    def __init__(
            self,
            max_lost=2,
            iou_threshold=0.5,
    ):
        self.iou_threshold = iou_threshold

        super(IOUTracker, self).__init__(max_lost=max_lost)

    def update(self, bboxes:list, detection_scores:list, class_ids:list) -> list:
        """
        Update the tracker based on the new bounding boxes.

        Args:
            bboxes (list): List of bounding boxes detected in the current frame. Each element of the list represent
                coordinates of bounding box as tuple `(top-left-x, top-left-y, width, height)`.
            detection_scores(list): List of detection scores (probability) of each detected object.
            class_ids (list): List of class_ids (int) corresponding to labels of the detected object.

        Returns:
            list: List of tracks being currently tracked by the tracker. Each track is represented by the tuple with elements `(frame_id, track_id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z)`.
        """
        detections = Tracker.preprocess_input(bboxes, class_ids, detection_scores)
        self.frame_count += 1
        track_ids = list(self.tracks.keys())

        updated_tracks = []
        for track_id in track_ids:
            if len(detections) > 0:
                idx, best_match = max(enumerate(detections), key=lambda x: iou(self.tracks[track_id].bbox, x[1][0]))
                (bb, cid, scr) = best_match

                if iou(self.tracks[track_id].bbox, bb) > self.iou_threshold:
                    self._update_track(track_id, self.frame_count, bb, scr, class_id=cid, center=get_centroid(bb),
                                       iou_score=iou(self.tracks[track_id].bbox, bb))
                    updated_tracks.append(track_id)
                    del detections[idx]

            if len(updated_tracks) == 0 or track_id is not updated_tracks[-1]:
                self.tracks[track_id].lost += 1
                if self.tracks[track_id].lost > self.max_lost:
                    self._remove_track(track_id)

        for bb, cid, scr in detections:
            self._add_track(self.frame_count, bb, scr, class_id=cid, center=get_centroid(bb))

        outputs = self._get_tracks(self.tracks)
        return outputs