from fastapi.testclient import TestClient

from signature_project.main import app

client = TestClient(app)


def verify(reference=None, candidate=None):
    files = {}
    if reference is not None:
        files["reference"] = reference
    if candidate is not None:
        files["candidate"] = candidate
    return client.post("/verify", files=files)


def test_accepts_two_images(png_bytes, jpg_bytes):
    response = verify(
        reference=("original.png", png_bytes),
        candidate=("check.jpg", jpg_bytes),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["reference_filename"] == "original.png"
    assert body["candidate_filename"] == "check.jpg"
    assert body["reference_size"] == [60, 30]


def test_missing_reference_returns_422(png_bytes):
    response = verify(candidate=("check.png", png_bytes))

    assert response.status_code == 422


def test_missing_candidate_returns_422(png_bytes):
    response = verify(reference=("original.png", png_bytes))

    assert response.status_code == 422


def test_pdf_is_not_accepted_as_reference(png_bytes, make_pdf):
    response = verify(
        reference=("original.pdf", make_pdf(pages=1)),
        candidate=("check.png", png_bytes),
    )

    assert response.status_code == 415
    assert response.json()["detail"].startswith("reference:")


def test_docx_is_not_accepted_as_candidate(png_bytes, make_docx):
    response = verify(
        reference=("original.png", png_bytes),
        candidate=("check.docx", make_docx(images=1)),
    )

    assert response.status_code == 415
    assert response.json()["detail"].startswith("candidate:")


def test_fake_candidate_is_rejected_and_named(png_bytes):
    response = verify(
        reference=("original.png", png_bytes),
        candidate=("check.png", b"just text pretending to be a png"),
    )

    assert response.status_code == 400
    assert response.json()["detail"].startswith("candidate:")


def test_empty_reference_is_rejected_and_named(png_bytes):
    response = verify(
        reference=("original.png", b""),
        candidate=("check.png", png_bytes),
    )

    assert response.status_code == 400
    assert response.json()["detail"].startswith("reference:")
