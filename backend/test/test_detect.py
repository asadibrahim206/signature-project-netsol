from fastapi.testclient import TestClient

from signature_project import validation
from signature_project.main import app

client = TestClient(app)


def upload(filename: str, content: bytes):
    return client.post("/detect", files={"file": (filename, content)})


def test_accepts_png(png_bytes):
    response = upload("sign.png", png_bytes)

    assert response.status_code == 200
    assert response.json()["file_type"] == "image"
    assert response.json()["filename"] == "sign.png"
    assert response.json()["images_extracted"] == 1


def test_accepts_jpg(jpg_bytes):
    response = upload("sign.jpg", jpg_bytes)

    assert response.status_code == 200
    assert response.json()["file_type"] == "image"


def test_accepts_pdf_and_counts_pages(make_pdf):
    response = upload("contract.pdf", make_pdf(pages=2))

    assert response.status_code == 200
    assert response.json()["file_type"] == "pdf"
    assert response.json()["images_extracted"] == 2


def test_accepts_docx_and_counts_images(make_docx):
    response = upload("agreement.docx", make_docx(images=3))

    assert response.status_code == 200
    assert response.json()["file_type"] == "docx"
    assert response.json()["images_extracted"] == 3


def test_extension_is_case_insensitive(png_bytes):
    response = upload("SIGN.PNG", png_bytes)

    assert response.status_code == 200


def test_rejects_unsupported_extension():
    response = upload("notes.txt", b"hello")

    assert response.status_code == 415


def test_rejects_empty_file():
    response = upload("sign.png", b"")

    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_rejects_fake_file_with_wrong_content():
    # A text file renamed to .png must not get through.
    response = upload("sign.png", b"this is just text")

    assert response.status_code == 400
    assert "does not match" in response.json()["detail"]


def test_rejects_corrupt_image_with_correct_header():
    # Passes the header check but cannot actually be decoded.
    response = upload("sign.png", b"\x89PNG\r\n\x1a\n" + b"garbage")

    assert response.status_code == 400
    assert "Could not read" in response.json()["detail"]


def test_rejects_file_that_is_too_large(monkeypatch, png_bytes):
    monkeypatch.setattr(validation, "MAX_FILE_SIZE_BYTES", 10)

    response = upload("sign.png", png_bytes)

    assert response.status_code == 413


def test_missing_file_returns_422():
    response = client.post("/detect")

    assert response.status_code == 422
