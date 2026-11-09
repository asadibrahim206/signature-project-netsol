import threading

from signature_project.detector import SignatureDetector
from signature_project.settings import (
    DETECT_CONF_THRESHOLD,
    DETECT_MODEL_PATH,
    DETECT_NMS_THRESHOLD,
    MODEL_DEVICE,
    VERIFY_MODEL_PATH,
)
from signature_project.verifier import SignatureVerifier

# Each worker process loads a model the first time it needs it, then keeps it in
# memory for every later job. (Loading takes seconds, so we never do it per job.)
_lock = threading.Lock()
_detector: SignatureDetector | None = None
_verifier: SignatureVerifier | None = None


def get_worker_detector() -> SignatureDetector:
    global _detector
    with _lock:
        if _detector is None:
            if not DETECT_MODEL_PATH.exists():
                raise RuntimeError(f"Detection model not found at {DETECT_MODEL_PATH}")
            _detector = SignatureDetector.load(
                DETECT_MODEL_PATH, DETECT_CONF_THRESHOLD, DETECT_NMS_THRESHOLD
            )
    return _detector


def get_worker_verifier() -> SignatureVerifier:
    global _verifier
    with _lock:
        if _verifier is None:
            if not VERIFY_MODEL_PATH.exists():
                raise RuntimeError(f"Verification model not found at {VERIFY_MODEL_PATH}")
            _verifier = SignatureVerifier.load(VERIFY_MODEL_PATH, MODEL_DEVICE)
    return _verifier
