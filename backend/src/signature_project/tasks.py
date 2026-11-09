from fastapi import HTTPException

from signature_project.celery_app import celery_app
from signature_project.detection_service import run_detection
from signature_project.job_store import CANDIDATE_SLOT, REFERENCE_SLOT, get_job_store
from signature_project.verification_service import run_verification
from signature_project.worker_models import get_worker_detector, get_worker_verifier


@celery_app.task(name="signature.detect")
def detect_signature_task(job_id: str, filename: str | None) -> dict:
    """Runs inside a WORKER process, not inside the web server.

    It returns a small envelope:
      {"ok": True,  "result": {...}}   the finished detection result
      {"ok": False, "error": "..."}    an expected problem, with a message safe to show the user
    Unexpected crashes are not caught here: Celery records them as a failed job,
    and the status endpoint shows a generic message (never internal details).
    """
    store = get_job_store()
    contents = store.load_upload(job_id)
    if contents is None:
        return {"ok": False, "error": "The uploaded file is no longer available. Please upload it again."}

    try:
        response = run_detection(get_worker_detector(), filename, contents)
        return {"ok": True, "result": response.model_dump()}
    except HTTPException as problem:       # e.g. a corrupt PDF
        return {"ok": False, "error": problem.detail}
    finally:
        store.delete_upload(job_id)         # the file is not needed any more


@celery_app.task(name="signature.verify")
def verify_signature_task(job_id: str, reference_filename: str | None, candidate_filename: str | None) -> dict:
    """Runs inside a WORKER process: compares the two uploaded signatures.

    Returns the same envelope as the detection task:
      {"ok": True,  "result": {...}}   the finished verification result
      {"ok": False, "error": "..."}    an expected problem, safe to show the user
    """
    store = get_job_store()
    reference = store.load_upload(job_id, REFERENCE_SLOT)
    candidate = store.load_upload(job_id, CANDIDATE_SLOT)
    if reference is None or candidate is None:
        store.delete_upload(job_id, REFERENCE_SLOT)
        store.delete_upload(job_id, CANDIDATE_SLOT)
        return {"ok": False, "error": "The uploaded files are no longer available. Please upload them again."}

    try:
        response = run_verification(
            get_worker_verifier(), reference_filename, reference, candidate_filename, candidate
        )
        return {"ok": True, "result": response.model_dump()}
    except HTTPException as problem:
        return {"ok": False, "error": problem.detail}
    finally:
        store.delete_upload(job_id, REFERENCE_SLOT)
        store.delete_upload(job_id, CANDIDATE_SLOT)
