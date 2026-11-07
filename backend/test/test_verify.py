import io

from fastapi.testclient import TestClient
from PIL import Image

from signature_project.main import app


def to_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def post_verify(client, reference=None, candidate=None):
    files = {}
    if reference is not None:
        files["reference"] = reference
    if candidate is not None:
        files["candidate"] = candidate
    return client.post("/verify", files=files)


# ---- the answer -------------------------------------------------------------------


def test_matching_signature(verify_client, fake_verifier, png_bytes, jpg_bytes):
    fake_verifier.score = 0.95

    response = post_verify(
        verify_client,
        reference=("original.png", png_bytes),
        candidate=("check.jpg", jpg_bytes),
    )

    assert response.status_code == 200
    assert response.json() == {
        "reference_filename": "original.png",
        "candidate_filename": "check.jpg",
        "score": 0.95,
        "threshold": 0.88,
        "is_genuine": True,
        "verdict": "The signature matches the reference.",
    }


def test_non_matching_signature(verify_client, fake_verifier, png_bytes):
    fake_verifier.score = 0.20

    response = post_verify(
        verify_client,
        reference=("original.png", png_bytes),
        candidate=("check.png", png_bytes),
    )

    body = response.json()
    assert response.status_code == 200
    assert body["is_genuine"] is False
    assert body["verdict"] == "The signature does not match the reference."


# ---- what the model receives ------------------------------------------------------


def test_reference_and_candidate_reach_the_model_in_order(verify_client, fake_verifier):
    reference = to_png(Image.new("RGB", (60, 30), "white"))
    candidate = to_png(Image.new("RGB", (80, 40), "white"))

    post_verify(
        verify_client,
        reference=("original.png", reference),
        candidate=("check.png", candidate),
    )

    sent_reference, sent_candidate = fake_verifier.calls[0]
    assert sent_reference.size == (60, 30)
    assert sent_candidate.size == (80, 40)


def test_transparent_png_keeps_its_transparency(verify_client, fake_verifier):
    transparent = to_png(Image.new("RGBA", (60, 30), (255, 255, 255, 0)))

    post_verify(
        verify_client,
        reference=("original.png", transparent),
        candidate=("check.png", transparent),
    )

    sent_reference, _ = fake_verifier.calls[0]
    assert sent_reference.mode == "RGBA"


# ---- bad input --------------------------------------------------------------------


def test_missing_reference_returns_422(verify_client, png_bytes):
    response = post_verify(verify_client, candidate=("check.png", png_bytes))

    assert response.status_code == 422


def test_missing_candidate_returns_422(verify_client, png_bytes):
    response = post_verify(verify_client, reference=("original.png", png_bytes))

    assert response.status_code == 422


def test_pdf_is_not_accepted_as_reference(verify_client, png_bytes, make_pdf):
    response = post_verify(
        verify_client,
        reference=("original.pdf", make_pdf(pages=1)),
        candidate=("check.png", png_bytes),
    )

    assert response.status_code == 415
    assert response.json()["detail"].startswith("reference:")


def test_docx_is_not_accepted_as_candidate(verify_client, png_bytes, make_docx):
    response = post_verify(
        verify_client,
        reference=("original.png", png_bytes),
        candidate=("check.docx", make_docx(images=1)),
    )

    assert response.status_code == 415
    assert response.json()["detail"].startswith("candidate:")


def test_unsupported_extension_only_lists_image_types(verify_client, png_bytes):
    response = post_verify(
        verify_client,
        reference=("original.png", png_bytes),
        candidate=("notes.txt", b"hello"),
    )

    detail = response.json()["detail"]
    assert response.status_code == 415
    assert detail.startswith("candidate:")
    assert ".png" in detail
    assert ".pdf" not in detail


def test_fake_candidate_is_rejected_and_named(verify_client, png_bytes):
    response = post_verify(
        verify_client,
        reference=("original.png", png_bytes),
        candidate=("check.png", b"just text pretending to be a png"),
    )

    assert response.status_code == 400
    assert response.json()["detail"].startswith("candidate:")


def test_empty_reference_is_rejected_and_named(verify_client, png_bytes):
    response = post_verify(
        verify_client,
        reference=("original.png", b""),
        candidate=("check.png", png_bytes),
    )

    assert response.status_code == 400
    assert response.json()["detail"].startswith("reference:")


# ---- model not loaded -------------------------------------------------------------


def test_returns_503_when_the_model_is_not_loaded(monkeypatch, png_bytes):
    monkeypatch.setattr(app.state, "verifier", None, raising=False)

    response = post_verify(
        TestClient(app),
        reference=("original.png", png_bytes),
        candidate=("check.png", png_bytes),
    )

    assert response.status_code == 503
    assert "not loaded" in response.json()["detail"]
