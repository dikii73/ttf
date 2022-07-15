import numpy as np
import cv2


def get_centroid(bboxes:np.ndarray) -> np.ndarray:
    """
    Calculate centroids for multiple bounding boxes.
    Args:
        bboxes (numpy.ndarray): Array of shape `(n, 4)` or of shape `(4,)` where
            each row contains `(xmin, ymin, width, height)`.
    Returns:
        numpy.ndarray: Centroid (x, y) coordinates of shape `(n, 2)` or `(2,)`.
    """

    one_bbox = False
    if len(bboxes.shape) == 1:
        one_bbox = True
        bboxes = bboxes[None, :]

    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    w, h = bboxes[:, 2], bboxes[:, 3]

    xc = xmin + 0.5*w
    yc = ymin + 0.5*h

    x = np.hstack([xc[:, None], yc[:, None]])

    if one_bbox:
        x = x.flatten()
    return x


def draw_tracks(image:np.ndarray, tracks:list) -> np.ndarray:
    """
    Draw on input image.
    Args:
        image (numpy.ndarray): image
        tracks (list): list of tracks to be drawn on the image.
    Returns:
        numpy.ndarray: image with the track-ids drawn on it.
    """
    min_age = 10 # min age for draw track 

    for trk in tracks:

        trk_id = trk[1]
        xmin = int(trk[2])
        ymin = int(trk[3])
        width = int(trk[4])
        height = int(trk[5])
        points_center = trk[7]
        age = trk[8]
           
        if age > min_age: 

            cv2.rectangle(image, (xmin, ymin), (xmin + width, ymin + height), (0,0,255), 2)

            if len(points_center) > 1:
                for i, point in enumerate(points_center):
                    cv2.circle(image, (point[0], point[1]), 4, (0, 0, 255), -1)
                    if i > 0:
                        cv2.line(image, (points_center[i-1][0], points_center[i-1][1]), (point[0], point[1]), (0, 0, 255), 2)
            else:
                cv2.circle(image, (points_center[0][0], points_center[0][1]), 4, (0, 0, 255), -1)
            
            xcentroid, ycentroid = int(xmin + 0.5*width), int(ymin + 0.5*height)

            text = "ID {}".format(trk_id)

            cv2.putText(image, text, (xcentroid - 10, ycentroid - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(image, (xcentroid, ycentroid), 4, (0, 255, 0), -1)

    return image


def xyxy2xywh(xyxy:np.ndarray) -> np.ndarray:
    """
    Convert bounding box coordinates from (xmin, ymin, xmax, ymax) format to (xmin, ymin, width, height).
    Args:
        xyxy (numpy.ndarray):
    Returns:
        numpy.ndarray: Bounding box coordinates (xmin, ymin, width, height).
    """

    if len(xyxy.shape) == 2:
        w, h = xyxy[:, 2] - xyxy[:, 0] + 1, xyxy[:, 3] - xyxy[:, 1] + 1
        xywh = np.concatenate((xyxy[:, 0:2], w[:, None], h[:, None]), axis=1)
        return xywh.astype("int")
    elif len(xyxy.shape) == 1:
        (left, top, right, bottom) = xyxy
        width = right - left + 1
        height = bottom - top + 1
        return np.array([left, top, width, height]).astype('int')
    else:
        raise ValueError("Input shape not compatible.")


def iou(bbox1:np.ndarray, bbox2:np.ndarray) -> float:
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array or list[floats]): Bounding box of length 4 containing
            ``(x-top-left, y-top-left, x-bottom-right, y-bottom-right)``.
        bbox2 (numpy.array or list[floats]): Bounding box of length 4 containing
            ``(x-top-left, y-top-left, x-bottom-right, y-bottom-right)``.
    Returns:
        float: intersection-over-onion of bbox1, bbox2.
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1), (x0_2, y0_2, x1_2, y1_2) = bbox1, bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0.0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    iou_ = size_intersection / size_union

    return iou_


def iou_xywh(bbox1:np.ndarray, bbox2:np.ndarray) -> float:
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array or list[floats]): bounding box of length 4 containing ``(x-top-left, y-top-left, width, height)``.
        bbox2 (numpy.array or list[floats]): bounding box of length 4 containing ``(x-top-left, y-top-left, width, height)``.
    Returns:
        float: intersection-over-onion of bbox1, bbox2.
    """
    bbox1 = bbox1[0], bbox1[1], bbox1[0]+bbox1[2], bbox1[1]+bbox1[3]
    bbox2 = bbox2[0], bbox2[1], bbox2[0]+bbox2[2], bbox2[1]+bbox2[3]

    iou_ = iou(bbox1, bbox2)

    return iou_
