import io
import zipfile

import pymupdf
import pytest
from fastapi.testclient import TestClient
from PIL import Image

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
