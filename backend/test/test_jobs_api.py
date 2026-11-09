import time

import pytest
import redis

from signature_project.jobs import GENERIC_FAILURE, build_job_status
from signature_project.tasks import detect_signature_task


def wait_for_job(client, job_id: str, timeout: float = 15.0) -> dict:
    """Ask for the job's status until it is finished (done or failed)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout} seconds, last status: {body['status']}")


JOB_ID = "b" * 32
META = {"kind": "detect", "filename": "scan.png", "created_at": "2026-11-09T10:00:00+00:00"}


def submit(client, filename: str, content: bytes):
    return client.post("/jobs/detect", files={"file": (filename, content)})


# ---- the happy path, through a real worker ----------------------------------------


def test_submitting_returns_a_ticket_straight_away(jobs_client, png_bytes):
    response = submit(jobs_client, "sign.png", png_bytes)

    body = response.json()
    assert response.status_code == 202
    assert len(body["job_id"]) == 32
    assert body["status"] == "queued"
    assert body["status_url"].endswith(f"/jobs/{body['job_id']}")


def test_the_worker_finishes_the_job_with_the_detection_result(jobs_client, fake_detector, one_box, png_bytes):
    fake_detector.boxes = [one_box]
    job_id = submit(jobs_client, "sign.png", png_bytes).json()["job_id"]

    body = wait_for_job(jobs_client, job_id)

    assert body["status"] == "done"
    assert body["kind"] == "detect"
    assert body["filename"] == "sign.png"
    assert body["error"] is None
    assert body["result"]["signature_detected"] is True
    assert body["result"]["total_signatures"] == 1
    assert body["result"]["results"][0]["detections"][0]["confidence"] == 0.8123


def test_a_pdf_job_is_checked_page_by_page(jobs_client, fake_detector, one_box, make_pdf):
    fake_detector.boxes_per_call = [[], [one_box]]
    job_id = submit(jobs_client, "contract.pdf", make_pdf(pages=2)).json()["job_id"]

    body = wait_for_job(jobs_client, job_id)

    assert body["status"] == "done"
    assert [page["signature_found"] for page in body["result"]["results"]] == [False, True]


def test_the_uploaded_file_is_removed_once_processed(jobs_client, job_store, png_bytes):
    job_id = submit(jobs_client, "sign.png", png_bytes).json()["job_id"]

    wait_for_job(jobs_client, job_id)

    assert job_store.load_upload(job_id) is None


def test_several_jobs_do_not_get_mixed_up(jobs_client, fake_detector, one_box, png_bytes, make_pdf):
    fake_detector.boxes = [one_box]
    first = submit(jobs_client, "one.png", png_bytes).json()["job_id"]
    second = submit(jobs_client, "two.pdf", make_pdf(pages=3)).json()["job_id"]

    result_second = wait_for_job(jobs_client, second)
    result_first = wait_for_job(jobs_client, first)

    assert result_first["filename"] == "one.png"
    assert result_first["result"]["images_checked"] == 1
    assert result_second["filename"] == "two.pdf"
    assert result_second["result"]["images_checked"] == 3


# ---- things going wrong -----------------------------------------------------------


def test_a_bad_file_is_refused_before_it_is_queued(jobs_client, job_store):
    response = submit(jobs_client, "notes.txt", b"hello")

    assert response.status_code == 415
    assert job_store._client.keys("job:*") == []       # nothing was created


def test_a_fake_image_is_refused_before_it_is_queued(jobs_client, job_store):
    response = submit(jobs_client, "sign.png", b"just text pretending to be a png")

    assert response.status_code == 400
    assert job_store._client.keys("job:*") == []


def test_a_corrupt_pdf_fails_with_a_clear_message(jobs_client):
    # Starts like a PDF so it passes the quick check, but cannot really be opened.
    job_id = submit(jobs_client, "broken.pdf", b"%PDF-1.4 garbage").json()["job_id"]

    body = wait_for_job(jobs_client, job_id)

    assert body["status"] == "failed"
    assert body["error"] == "Could not read the PDF file."
    assert body["result"] is None


def test_an_unexpected_crash_shows_a_generic_message(jobs_client, fake_detector, png_bytes):
    def explode(image):
        raise RuntimeError("secret internal detail")

    fake_detector.detect = explode
    job_id = submit(jobs_client, "sign.png", png_bytes).json()["job_id"]

    body = wait_for_job(jobs_client, job_id)

    assert body["status"] == "failed"
    assert body["error"] == GENERIC_FAILURE
    assert "secret" not in str(body)


def test_a_job_whose_upload_expired_fails_cleanly(job_store, jobs_client):
    # The job record exists but the file is gone (it expired before a worker got to it).
    job_store.register(JOB_ID, "detect", "late.png")

    envelope = detect_signature_task.apply(args=[JOB_ID, "late.png"]).get()

    assert envelope["ok"] is False
    assert "no longer available" in envelope["error"]


def test_unknown_job_is_404(jobs_client):
    response = jobs_client.get(f"/jobs/{JOB_ID}")

    assert response.status_code == 404


def test_malformed_job_id_is_422(jobs_client):
    assert jobs_client.get("/jobs/not-a-valid-id").status_code == 422


def test_queue_down_gives_503_and_leaves_nothing_behind(jobs_client, job_store, monkeypatch, png_bytes):
    def redis_is_down(*args, **kwargs):
        raise redis.ConnectionError("connection refused")

    monkeypatch.setattr(job_store, "save_upload", redis_is_down)

    response = submit(jobs_client, "sign.png", png_bytes)

    assert response.status_code == 503
    assert "not available" in response.json()["detail"]


# ---- the status words -------------------------------------------------------------


@pytest.mark.parametrize(
    "state, expected",
    [("PENDING", "queued"), ("RECEIVED", "queued"), ("STARTED", "running"),
     ("RETRY", "running"), ("FAILURE", "failed"), ("REVOKED", "failed")],
)
def test_celery_states_become_simple_words(state, expected):
    assert build_job_status(JOB_ID, META, state, None).status == expected


def test_finished_job_shows_its_result():
    result = {
        "filename": "scan.png", "file_type": "image", "images_checked": 1,
        "signature_detected": False, "total_signatures": 0, "confidence_threshold": 0.5,
        "results": [], "verdict": "No signature detected.",
    }

    status = build_job_status(JOB_ID, META, "SUCCESS", {"ok": True, "result": result})

    assert status.status == "done"
    assert status.result.verdict == "No signature detected."


def test_expected_problem_shows_its_message():
    status = build_job_status(JOB_ID, META, "SUCCESS", {"ok": False, "error": "Could not read the PDF file."})

    assert status.status == "failed"
    assert status.error == "Could not read the PDF file."


def test_crash_never_leaks_details():
    status = build_job_status(JOB_ID, META, "FAILURE", RuntimeError("secret"))

    assert status.error == GENERIC_FAILURE
