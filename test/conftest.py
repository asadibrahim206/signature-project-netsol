import io
import zipfile

import fitz
import pytest
from PIL import Image


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
        document = fitz.open()
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
