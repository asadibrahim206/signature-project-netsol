from pathlib import Path

from fastapi import HTTPException, Request
from PIL import Image

from signature_project.detection_processing import DetectedBox, postprocess, preprocess

DEFAULT_INPUT_SIZE = (640, 640)
VALUES_PER_BOX = 6  # centre x, centre y, width, height, objectness, signature score


class SignatureDetector:
    """Wraps the ONNX detection model. The rest of the app never touches onnxruntime."""

    def __init__(self, session, conf_threshold: float = 0.5, nms_threshold: float = 0.45):
        self._session = session
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        model_input = session.get_inputs()[0]
        self._input_name = model_input.name
        self.input_size = self._read_input_size(model_input.shape)

        # Fail early, at startup, if someone drops the wrong model file in here.
        output_shape = session.get_outputs()[0].shape
        if output_shape[-1] != VALUES_PER_BOX:
            raise ValueError(
                f"Unexpected detection model: each box should have {VALUES_PER_BOX} values "
                f"(one class, decoded output) but the model gives {output_shape[-1]}."
            )

    @staticmethod
    def _read_input_size(shape) -> tuple[int, int]:
        """The model file says its own input size, e.g. [1, 3, 640, 640]."""
        height, width = shape[2], shape[3]
        if isinstance(height, int) and isinstance(width, int):
            return (height, width)
        return DEFAULT_INPUT_SIZE

    @classmethod
    def load(
        cls,
        model_path: Path,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.45,
    ) -> "SignatureDetector":
        # Imported here so the app can start (and tests can run) without paying
        # for onnxruntime unless a model is really being loaded.
        import onnxruntime as ort

        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        return cls(session, conf_threshold, nms_threshold)

    def detect(self, image: Image.Image) -> list[DetectedBox]:
        tensor, ratio = preprocess(image, self.input_size)
        output = self._session.run(None, {self._input_name: tensor})[0]
        return postprocess(output, ratio, image.size, self.conf_threshold, self.nms_threshold)


def get_detector(request: Request) -> SignatureDetector:
    """FastAPI dependency: hands the loaded model to an endpoint, or answers 503."""
    detector = getattr(request.app.state, "detector", None)
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="The detection model is not loaded.",
        )
    return detector
