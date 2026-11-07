import io
import zipfile

import pytest
from fastapi import HTTPException

from signature_project import extraction
from signature_project.extraction import extract_images


def test_image_gives_one_rgb_image(png_bytes):
    images = extract_images("image", png_bytes)

    assert len(images) == 1
    assert images[0].mode == "RGB"


def test_jpg_gives_one_image(jpg_bytes):
    assert len(extract_images("image", jpg_bytes)) == 1


def test_pdf_gives_one_image_per_page(make_pdf):
    images = extract_images("pdf", make_pdf(pages=3))

    assert len(images) == 3
    assert all(image.mode == "RGB" for image in images)


def test_pdf_with_too_many_pages_is_rejected(make_pdf, monkeypatch):
    monkeypatch.setattr(extraction, "MAX_PDF_PAGES", 2)

    with pytest.raises(HTTPException) as error:
        extract_images("pdf", make_pdf(pages=3))

    assert error.value.status_code == 400


def test_docx_gives_its_embedded_images(make_docx):
    images = extract_images("docx", make_docx(images=2))

    assert len(images) == 2


def test_docx_without_images_gives_empty_list(make_docx):
    assert extract_images("docx", make_docx(images=0)) == []


def test_zip_that_is_not_word_is_rejected():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("hello.txt", "not a word file")

    with pytest.raises(HTTPException) as error:
        extract_images("docx", buffer.getvalue())

    assert error.value.status_code == 400


def test_broken_image_is_rejected():
    with pytest.raises(HTTPException) as error:
        extract_images("image", b"\x89PNG\r\n\x1a\n" + b"garbage")

    assert error.value.status_code == 400


def test_broken_pdf_is_rejected():
    with pytest.raises(HTTPException) as error:
        extract_images("pdf", b"%PDF-1.4 garbage")

    assert error.value.status_code == 400
