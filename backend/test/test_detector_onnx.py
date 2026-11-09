"""Tests that run the detector through the real onnxruntime.

Since the trained detection model is not available while building, a tiny
STAND-IN model is created here. It ignores the picture and always answers with
the same known boxes. That proves everything around the model works: loading,
reading its input size, running it, decoding, duplicate removal, and the full
trip through the running app. It says nothing about detection ACCURACY.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from signature_project import main  # noqa: E402
from signature_project.detector import SignatureDetector  # noqa: E402
from signature_project.settings import DETECT_MODEL_PATH  # noqa: E402


@pytest.fixture(scope="module")
def stub_model_path(tmp_path_factory):
    rows = np.zeros((1, 8400, 6), dtype=np.float32)
    rows[0, 0] = [320, 320, 200, 80, 0.9, 0.9]    # score 0.81: a real signature
    rows[0, 1] = [322, 322, 200, 80, 0.9, 0.8]    # score 0.72: duplicate of the first
    rows[0, 2] = [100, 500, 60, 30, 0.2, 0.2]     # score 0.04: noise

    answer = helper.make_node(
        "Constant", [], ["output"], value=numpy_helper.from_array(rows, name="answer")
    )
    graph = helper.make_graph(
        [answer],
        "stub_detector",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8400, 6])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9  # an older format version that every onnxruntime understands

    path = tmp_path_factory.mktemp("models") / "stub_detector.onnx"
    onnx.save(model, path)
    return path


def test_model_file_tells_us_its_input_size(stub_model_path):
    detector = SignatureDetector.load(stub_model_path)

    assert detector.input_size == (640, 640)


def test_detector_finds_the_box_and_removes_the_duplicate(stub_model_path):
    detector = SignatureDetector.load(stub_model_path)
    page = Image.new("RGB", (1280, 1280), "white")   # ratio 0.5 -> boxes doubled

    boxes = detector.detect(page)

    assert len(boxes) == 1
    box = boxes[0]
    assert (box.x_min, box.y_min, box.x_max, box.y_max) == (440.0, 560.0, 840.0, 720.0)
    assert abs(box.confidence - 0.81) < 1e-6


def test_a_higher_confidence_threshold_removes_the_box(stub_model_path):
    detector = SignatureDetector.load(stub_model_path, conf_threshold=0.95)

    assert detector.detect(Image.new("RGB", (1280, 1280), "white")) == []


def test_a_model_with_the_wrong_output_shape_is_refused(tmp_path):
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["output"],
                          value=numpy_helper.from_array(np.zeros((1, 100, 85), np.float32)))],
        "wrong_detector",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 100, 85])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    path = tmp_path / "wrong.onnx"
    onnx.save(model, path)

    with pytest.raises(ValueError, match="Unexpected detection model"):
        SignatureDetector.load(path)


def test_whole_app_loads_the_model_at_startup_and_serves_detections(
    stub_model_path, tmp_path, monkeypatch, png_bytes
):
    # Point the app at the stand-in model, and make sure the verification model
    # (which needs torch and takes seconds to load) is not touched.
    monkeypatch.setattr(main, "DETECT_MODEL_PATH", stub_model_path)
    monkeypatch.setattr(main, "VERIFY_MODEL_PATH", tmp_path / "no_such_model.pt")

    with TestClient(main.app) as client:     # `with` runs the startup code
        response = client.post("/detect", files={"file": ("sign.png", png_bytes)})

    body = response.json()
    assert response.status_code == 200
    assert body["signature_detected"] is True
    assert body["total_signatures"] == 1
    assert body["results"][0]["detections"][0]["confidence"] == 0.81


# ---- the real trained model (skipped until models/signature_detector.onnx exists) --


@pytest.mark.skipif(
    not DETECT_MODEL_PATH.exists(),
    reason=f"detection model not found at {DETECT_MODEL_PATH}",
)
def test_real_model_loads_and_runs_on_a_blank_page():
    detector = SignatureDetector.load(DETECT_MODEL_PATH)

    boxes = detector.detect(Image.new("RGB", (1241, 1754), "white"))

    # Plumbing check only: whatever it returns must be well-formed.
    for box in boxes:
        assert 0.0 <= box.confidence <= 1.0
        assert box.x_max > box.x_min and box.y_max > box.y_min
