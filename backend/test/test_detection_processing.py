"""Tests for the maths around the model: getting a picture in, and boxes out.

Every number here was worked out by hand, so a failing test means the code
(not the test) is wrong.
"""
import numpy as np
from PIL import Image

from signature_project.detection_processing import PAD_VALUE, nms, postprocess, preprocess


def empty_output(rows: int = 8400) -> np.ndarray:
    return np.zeros((1, rows, 6), dtype=np.float32)


# ---- preprocess -------------------------------------------------------------------


def test_picture_is_shrunk_to_fit_and_kept_top_left():
    image = Image.new("RGB", (1280, 640), (0, 0, 0))  # wide picture, black

    tensor, ratio = preprocess(image, (640, 640))

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert ratio == 0.5
    assert tensor[0, 0, 0, 0] == 0                  # top-left: the picture (black)
    assert tensor[0, 0, 639, 0] == PAD_VALUE        # bottom: grey padding
    assert tensor[0, 0, 100, 639] == 0              # picture fills the full width


def test_small_picture_is_enlarged():
    tensor, ratio = preprocess(Image.new("RGB", (320, 160)), (640, 640))

    assert ratio == 2.0
    assert tensor.shape == (1, 3, 640, 640)


def test_colours_are_in_bgr_order():
    red = Image.new("RGB", (64, 64), (255, 0, 0))

    tensor, _ = preprocess(red, (640, 640))

    assert tensor[0, 0, 0, 0] == 0      # channel 0 is blue
    assert tensor[0, 2, 0, 0] == 255    # channel 2 is red


def test_non_rgb_pictures_are_accepted():
    transparent = Image.new("RGBA", (100, 50), (0, 0, 0, 0))
    grey = Image.new("L", (100, 50), 128)

    assert preprocess(transparent, (640, 640))[0].shape == (1, 3, 640, 640)
    assert preprocess(grey, (640, 640))[0].shape == (1, 3, 640, 640)


def test_extremely_thin_picture_does_not_crash():
    tensor, _ = preprocess(Image.new("RGB", (5000, 1)), (640, 640))

    assert tensor.shape == (1, 3, 640, 640)


# ---- postprocess ------------------------------------------------------------------


def test_box_is_mapped_back_to_the_original_picture():
    output = empty_output()
    # centre (320, 160), 200 wide, 100 tall; confidence 0.9 * 0.9 = 0.81
    output[0, 0] = [320, 160, 200, 100, 0.9, 0.9]

    boxes = postprocess(output, ratio=0.5, image_size=(1280, 1280), conf_threshold=0.5, nms_threshold=0.45)

    assert len(boxes) == 1
    box = boxes[0]
    assert (box.x_min, box.y_min, box.x_max, box.y_max) == (440.0, 220.0, 840.0, 420.0)
    assert abs(box.confidence - 0.81) < 1e-6


def test_low_confidence_boxes_are_dropped():
    output = empty_output()
    output[0, 0] = [320, 160, 200, 100, 0.4, 0.5]  # 0.4 * 0.5 = 0.2

    assert postprocess(output, 1.0, (640, 640), 0.5, 0.45) == []


def test_threshold_decides_what_is_kept():
    output = empty_output()
    output[0, 0] = [320, 160, 200, 100, 0.9, 0.9]  # 0.81

    assert len(postprocess(output, 1.0, (640, 640), 0.8, 0.45)) == 1
    assert len(postprocess(output, 1.0, (640, 640), 0.9, 0.45)) == 0


def test_overlapping_duplicates_keep_only_the_best():
    output = empty_output()
    output[0, 0] = [320, 160, 200, 100, 0.9, 0.9]   # 0.81
    output[0, 1] = [324, 162, 200, 100, 0.9, 0.7]   # 0.63, almost the same box

    boxes = postprocess(output, 1.0, (640, 640), 0.5, 0.45)

    assert len(boxes) == 1
    assert abs(boxes[0].confidence - 0.81) < 1e-6


def test_separate_signatures_are_both_kept_best_first():
    output = empty_output()
    output[0, 0] = [100, 100, 80, 40, 0.9, 0.7]   # 0.63
    output[0, 1] = [500, 500, 80, 40, 0.9, 0.9]   # 0.81

    boxes = postprocess(output, 1.0, (640, 640), 0.5, 0.45)

    assert len(boxes) == 2
    assert boxes[0].confidence > boxes[1].confidence


def test_boxes_are_trimmed_to_the_picture():
    output = empty_output()
    output[0, 0] = [600, 320, 200, 100, 0.9, 0.9]   # sticks out past x = 640

    box = postprocess(output, 1.0, (640, 640), 0.5, 0.45)[0]

    assert box.x_max == 640.0
    assert box.x_min == 500.0


def test_empty_output_gives_no_boxes():
    assert postprocess(empty_output(), 1.0, (640, 640), 0.5, 0.45) == []


# ---- nms --------------------------------------------------------------------------


def test_nms_suppresses_only_heavily_overlapping_boxes():
    boxes = np.array(
        [[0, 0, 100, 100], [5, 5, 105, 105], [300, 300, 400, 400]],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    assert nms(boxes, scores, 0.45) == [0, 2]
