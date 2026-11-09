import io
import zipfile

import fakeredis
import pymupdf
import pytest
from celery.contrib.testing.worker import start_worker
from fastapi.testclient import TestClient
from PIL import Image

from signature_project import tasks
from signature_project.celery_app import celery_app
from signature_project.job_store import JobStore, get_job_store

from signature_project.detection_processing import DetectedBox
from signature_project.detector import get_detector
from signature_project.main import app
from signature_project.verifier import VerificationResult, get_verifier


def _image_bytes(fmt: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (60, 30), "white").save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def png_bytes() -> bytes:
    return _image_bytes("PNG")


@pytest.fixture
def jpg_bytes() -> bytes:
    return _image_bytes("JPEG")


@pytest.fixture
def make_pdf():
    """Returns a function that builds a real PDF with the given number of pages."""

    def build(pages: int = 1) -> bytes:
        document = pymupdf.open()
        for number in range(pages):
            page = document.new_page()
            page.insert_text((72, 72), f"Page {number + 1}")
        data = document.tobytes()
        document.close()
        return data

    return build


@pytest.fixture
def make_docx(png_bytes):
    """Returns a function that builds a minimal Word file with N embedded images."""

    def build(images: int = 1) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("word/document.xml", "<w:document/>")
            for number in range(images):
                archive.writestr(f"word/media/image{number + 1}.png", png_bytes)
        return buffer.getvalue()

    return build


class FakeVerifier:
    """Stands in for the real model in endpoint tests: instant, and we control the answer."""

    def __init__(self, score: float = 0.95, threshold: float = 0.88):
        self.score = score
        self.threshold = threshold
        self.calls = []  # every (reference, candidate) pair it was asked to compare

    def verify(self, reference, candidate):
        self.calls.append((reference, candidate))
        return VerificationResult(
            score=self.score,
            threshold=self.threshold,
            is_genuine=self.score >= self.threshold,
        )


@pytest.fixture
def fake_verifier():
    return FakeVerifier()


@pytest.fixture
def verify_client(fake_verifier):
    """A test client whose /verify uses the fake model instead of the real one."""
    app.dependency_overrides[get_verifier] = lambda: fake_verifier
    yield TestClient(app)
    app.dependency_overrides.clear()


class FakeDetector:
    """Stands in for the real detection model in endpoint tests.

    By default every picture gets the same boxes. Pass `boxes_per_call` to give
    each picture (first call, second call, ...) its own list of boxes.
    """

    conf_threshold = 0.5

    def __init__(self, boxes=None, boxes_per_call=None):
        self.boxes = boxes or []
        self.boxes_per_call = boxes_per_call
        self.calls = []  # every picture it was asked to check

    def detect(self, image):
        index = len(self.calls)
        self.calls.append(image)
        if self.boxes_per_call is not None:
            return self.boxes_per_call[index]
        return self.boxes


@pytest.fixture
def one_box():
    return DetectedBox(x_min=10.0, y_min=5.0, x_max=50.0, y_max=25.0, confidence=0.8123)


@pytest.fixture
def fake_detector():
    return FakeDetector()


@pytest.fixture
def detect_client(fake_detector):
    """A test client whose /detect uses the fake detector instead of the real model."""
    app.dependency_overrides[get_detector] = lambda: fake_detector
    yield TestClient(app)
    app.dependency_overrides.clear()


# ---- background jobs --------------------------------------------------------------


@pytest.fixture(scope="session")
def queue_worker():
    """A real Celery worker running in a background thread of the test process.

    The waiting line is in memory (no Redis needed), but the jobs travel the very same
    path as in production: submitted as messages, picked up by a worker, result stored.
    """
    celery_app.conf.update(broker_url="memory://", result_backend="cache+memory://")
    with start_worker(celery_app, perform_ping_check=False, pool="solo", loglevel="ERROR"):
        yield celery_app


@pytest.fixture
def job_store():
    """A job store backed by fakeredis, an in-memory stand-in that behaves like Redis."""
    return JobStore(fakeredis.FakeRedis(), ttl_seconds=60)


@pytest.fixture
def jobs_client(queue_worker, job_store, fake_detector, monkeypatch):
    """A test client plus a running worker, both using the fake store and fake detector."""
    app.dependency_overrides[get_job_store] = lambda: job_store
    monkeypatch.setattr(tasks, "get_job_store", lambda: job_store)           # what the worker uses
    monkeypatch.setattr(tasks, "get_worker_detector", lambda: fake_detector)
    yield TestClient(app)
    app.dependency_overrides.clear()

