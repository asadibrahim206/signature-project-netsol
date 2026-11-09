from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

# Grey padding value. YOLOX used 114 while training, so we must use it too.
PAD_VALUE = 114


@dataclass(frozen=True)
class DetectedBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float


def preprocess(image: Image.Image, input_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Turn a picture into the exact array the YOLOX model expects.

    Steps (the same as YOLOX's own code during training):
    1. Shrink or enlarge the picture to fit inside the model's square, keeping its shape.
    2. Place it in the top-left corner of a grey square.
    3. Use BGR colour order and channels-first layout, as plain 0-255 floats.

    Returns the array and the resize ratio, which we need later to map the boxes
    back onto the original picture.
    """
    input_h, input_w = input_size
    rgb = np.asarray(image.convert("RGB"))
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])

    height, width = bgr.shape[:2]
    ratio = min(input_h / height, input_w / width)
    new_w = max(1, int(width * ratio))
    new_h = max(1, int(height * ratio))

    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_h, input_w, 3), PAD_VALUE, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized

    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
    return tensor, ratio


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Non-maximum suppression: when boxes overlap too much, keep only the best one."""
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size > 0:
        best = order[0]
        keep.append(int(best))
        rest = order[1:]
        inter_w = np.minimum(boxes[best, 2], boxes[rest, 2]) - np.maximum(boxes[best, 0], boxes[rest, 0])
        inter_h = np.minimum(boxes[best, 3], boxes[rest, 3]) - np.maximum(boxes[best, 1], boxes[rest, 1])
        inter = np.clip(inter_w, 0, None) * np.clip(inter_h, 0, None)
        iou = inter / (areas[best] + areas[rest] - inter + 1e-9)
        order = rest[iou <= threshold]
    return keep


def postprocess(
    output: np.ndarray,
    ratio: float,
    image_size: tuple[int, int],
    conf_threshold: float,
    nms_threshold: float,
) -> list[DetectedBox]:
    """Model output -> clean list of boxes in the ORIGINAL picture's pixels.

    `output` is the decoded YOLOX output: one row per candidate box, with
    (centre x, centre y, width, height, objectness, class score).
    `image_size` is (width, height) of the original picture.
    """
    predictions = output[0] if output.ndim == 3 else output
    scores = predictions[:, 4] * predictions[:, 5]

    keep = scores >= conf_threshold
    predictions, scores = predictions[keep], scores[keep]
    if scores.size == 0:
        return []

    cx, cy, w, h = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1) / ratio

    width, height = image_size

    # Same order as YOLOX: remove duplicates first, THEN trim boxes to the picture.
    detections = []
    for index in nms(boxes, scores, nms_threshold):
        x_min, y_min, x_max, y_max = (float(v) for v in boxes[index])
        x_min, x_max = min(max(x_min, 0.0), width), min(max(x_max, 0.0), width)
        y_min, y_max = min(max(y_min, 0.0), height), min(max(y_max, 0.0), height)
        if x_max > x_min and y_max > y_min:   # drop boxes squashed flat by trimming
            detections.append(DetectedBox(x_min, y_min, x_max, y_max, float(scores[index])))
    return detections
