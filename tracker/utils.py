import numpy as np
import cv2

def get_centroid(bboxes):
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

def draw_tracks(image, tracks):
    """
    Draw on input image.
    Args:
        image (numpy.ndarray): image
        tracks (list): list of tracks to be drawn on the image.
    Returns:
        numpy.ndarray: image with the track-ids drawn on it.
    """

    for trk in tracks:

        trk_id = trk[1]
        xmin = trk[2]
        ymin = trk[3]
        width = trk[4]
        height = trk[5]

        xcentroid, ycentroid = int(xmin + 0.5*width), int(ymin + 0.5*height)

        text = "ID {}".format(trk_id)

        cv2.putText(image, text, (xcentroid - 10, ycentroid - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(image, (xcentroid, ycentroid), 4, (0, 255, 0), -1)

    return image

def xyxy2xywh(xyxy):
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
