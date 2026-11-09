from fastapi.testclient import TestClient

from signature_project import validation
from signature_project.main import app


def upload(client, filename: str, content: bytes):
    return client.post("/detect", files={"file": (filename, content)})


# ---- the answer -------------------------------------------------------------------


def test_signature_found_in_an_image(detect_client, fake_detector, one_box, png_bytes):
    fake_detector.boxes = [one_box]

    response = upload(detect_client, "sign.png", png_bytes)

    assert response.status_code == 200
    assert response.json() == {
        "filename": "sign.png",
        "file_type": "image",
        "images_checked": 1,
        "signature_detected": True,
        "total_signatures": 1,
        "confidence_threshold": 0.5,
        "results": [
            {
                "page": 1,
                "width": 60,
                "height": 30,
                "signature_found": True,
                "detections": [
                    {
                        "confidence": 0.8123,
                        "box": {"x_min": 10.0, "y_min": 5.0, "x_max": 50.0, "y_max": 25.0},
                    }
                ],
            }
        ],
        "verdict": "Signature detected (1 found).",
    }


def test_no_signature_found(detect_client, png_bytes):
    response = upload(detect_client, "blank.png", png_bytes)

    body = response.json()
    assert response.status_code == 200
    assert body["signature_detected"] is False
    assert body["total_signatures"] == 0
    assert body["results"][0]["detections"] == []
    assert body["verdict"] == "No signature detected."


def test_pdf_is_checked_page_by_page(detect_client, fake_detector, one_box, make_pdf):
    fake_detector.boxes_per_call = [[], [one_box]]  # signature only on page 2

    response = upload(detect_client, "contract.pdf", make_pdf(pages=2))

    body = response.json()
    assert response.status_code == 200
    assert body["file_type"] == "pdf"
    assert body["images_checked"] == 2
    assert [page["signature_found"] for page in body["results"]] == [False, True]
    assert [page["page"] for page in body["results"]] == [1, 2]
    assert body["total_signatures"] == 1


def test_several_signatures_are_counted(detect_client, fake_detector, one_box, png_bytes):
    fake_detector.boxes = [one_box, one_box]

    response = upload(detect_client, "sign.png", png_bytes)

    assert response.json()["total_signatures"] == 2
    assert response.json()["verdict"] == "Signature detected (2 found)."


def test_docx_images_are_all_checked(detect_client, make_docx):
    response = upload(detect_client, "agreement.docx", make_docx(images=3))

    assert response.status_code == 200
    assert response.json()["file_type"] == "docx"
    assert response.json()["images_checked"] == 3


def test_docx_without_images_says_so(detect_client, make_docx):
    response = upload(detect_client, "agreement.docx", make_docx(images=0))

    body = response.json()
    assert response.status_code == 200
    assert body["images_checked"] == 0
    assert body["signature_detected"] is False
    assert "No images were found" in body["verdict"]


def test_detector_receives_the_uploaded_picture(detect_client, fake_detector, png_bytes):
    upload(detect_client, "sign.png", png_bytes)

    assert fake_detector.calls[0].size == (60, 30)


def test_extension_is_case_insensitive(detect_client, png_bytes):
    assert upload(detect_client, "SIGN.PNG", png_bytes).status_code == 200


# ---- bad input --------------------------------------------------------------------


def test_rejects_unsupported_extension(detect_client):
    response = upload(detect_client, "notes.txt", b"hello")

    assert response.status_code == 415


def test_rejects_empty_file(detect_client):
    response = upload(detect_client, "sign.png", b"")

    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_rejects_fake_file_with_wrong_content(detect_client):
    # A text file renamed to .png must not get through.
    response = upload(detect_client, "sign.png", b"this is just text")

    assert response.status_code == 400
    assert "does not match" in response.json()["detail"]


def test_rejects_corrupt_image_with_correct_header(detect_client):
    # Passes the header check but cannot actually be decoded.
    response = upload(detect_client, "sign.png", b"\x89PNG\r\n\x1a\n" + b"garbage")

    assert response.status_code == 400
    assert "Could not read" in response.json()["detail"]


def test_rejects_file_that_is_too_large(detect_client, monkeypatch, png_bytes):
    monkeypatch.setattr(validation, "MAX_FILE_SIZE_BYTES", 10)

    response = upload(detect_client, "sign.png", png_bytes)

    assert response.status_code == 413


def test_missing_file_returns_422(detect_client):
    response = detect_client.post("/detect")

    assert response.status_code == 422


# ---- model not loaded -------------------------------------------------------------


def test_returns_503_when_the_model_is_not_loaded(monkeypatch, png_bytes):
    monkeypatch.setattr(app.state, "detector", None, raising=False)

    response = upload(TestClient(app), "sign.png", png_bytes)

    assert response.status_code == 503
    assert "not loaded" in response.json()["detail"]
