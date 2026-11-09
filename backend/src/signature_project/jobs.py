import uuid

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, File, HTTPException, Path, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from kombu.exceptions import OperationalError
from redis.exceptions import RedisError

from signature_project.celery_app import celery_app
from signature_project.job_store import CANDIDATE_SLOT, REFERENCE_SLOT, JobStore, get_job_store
from signature_project.schemas import JobStatusResponse, JobSubmitResponse
from signature_project.tasks import detect_signature_task, verify_signature_task
from signature_project.validation import validate_upload
from signature_project.verification_service import read_signature_image

router = APIRouter(tags=["Jobs"])

JOB_ID_PATTERN = r"^[0-9a-f]{32}$"
QUEUE_UNAVAILABLE = "The job queue is not available right now. Please try again shortly."
GENERIC_FAILURE = "Processing failed unexpectedly. Please try again."

# Celery's state names -> the simple words we show to API users.
RUNNING_STATES = {"STARTED", "RETRY"}
FAILED_STATES = {"FAILURE", "REVOKED"}


def build_job_status(job_id: str, meta: dict, state: str, payload) -> JobStatusResponse:
    """Turn what Celery knows about a job into our simple status answer."""
    base = dict(
        job_id=job_id,
        kind=meta["kind"],
        filename=meta.get("filename"),
        created_at=meta["created_at"],
    )
    if state == "SUCCESS" and isinstance(payload, dict):
        if payload.get("ok"):
            return JobStatusResponse(**base, status="done", result=payload["result"])
        return JobStatusResponse(**base, status="failed", error=payload.get("error", GENERIC_FAILURE))
    if state in FAILED_STATES:
        return JobStatusResponse(**base, status="failed", error=GENERIC_FAILURE)
    if state in RUNNING_STATES:
        return JobStatusResponse(**base, status="running")
    return JobStatusResponse(**base, status="queued")   # PENDING / RECEIVED: waiting in line


def _enqueue_detection(store: JobStore, job_id: str, filename: str | None, contents: bytes) -> None:
    store.register(job_id, "detect", filename)
    store.save_upload(job_id, contents)
    try:
        detect_signature_task.apply_async(args=[job_id, filename], task_id=job_id)
    except Exception:
        store.delete_upload(job_id)     # do not leave an orphaned file behind
        raise


def _enqueue_verification(
    store: JobStore,
    job_id: str,
    reference_filename: str | None,
    reference: bytes,
    candidate_filename: str | None,
    candidate: bytes,
) -> None:
    store.register(job_id, "verify", f"{reference_filename} vs {candidate_filename}")
    store.save_upload(job_id, reference, REFERENCE_SLOT)
    store.save_upload(job_id, candidate, CANDIDATE_SLOT)
    try:
        verify_signature_task.apply_async(
            args=[job_id, reference_filename, candidate_filename], task_id=job_id
        )
    except Exception:
        store.delete_upload(job_id, REFERENCE_SLOT)     # do not leave orphaned files behind
        store.delete_upload(job_id, CANDIDATE_SLOT)
        raise


@router.post("/jobs/detect", status_code=202, response_model=JobSubmitResponse)
async def submit_detection_job(
    request: Request,
    file: UploadFile = File(...),
    store: JobStore = Depends(get_job_store),
):
    contents = await file.read()
    # Check the file NOW, so a bad upload is refused immediately instead of
    # waiting in line only to fail later.
    validate_upload(file.filename, contents)

    job_id = uuid.uuid4().hex
    try:
        await run_in_threadpool(_enqueue_detection, store, job_id, file.filename, contents)
    except (RedisError, OperationalError):
        raise HTTPException(status_code=503, detail=QUEUE_UNAVAILABLE)

    return JobSubmitResponse(
        job_id=job_id,
        status="queued",
        status_url=str(request.url_for("job_status", job_id=job_id)),
    )


@router.post("/jobs/verify", status_code=202, response_model=JobSubmitResponse)
async def submit_verification_job(
    request: Request,
    reference: UploadFile = File(..., description="The original, trusted signature"),
    candidate: UploadFile = File(..., description="The signature to check"),
    store: JobStore = Depends(get_job_store),
):
    reference_bytes = await reference.read()
    candidate_bytes = await candidate.read()
    # Check both files NOW, so a bad upload is refused immediately instead of
    # waiting in line only to fail later.
    await run_in_threadpool(read_signature_image, reference.filename, reference_bytes, "reference")
    await run_in_threadpool(read_signature_image, candidate.filename, candidate_bytes, "candidate")

    job_id = uuid.uuid4().hex
    try:
        await run_in_threadpool(
            _enqueue_verification,
            store,
            job_id,
            reference.filename,
            reference_bytes,
            candidate.filename,
            candidate_bytes,
        )
    except (RedisError, OperationalError):
        raise HTTPException(status_code=503, detail=QUEUE_UNAVAILABLE)

    return JobSubmitResponse(
        job_id=job_id,
        status="queued",
        status_url=str(request.url_for("job_status", job_id=job_id)),
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(
    job_id: str = Path(..., pattern=JOB_ID_PATTERN),
    store: JobStore = Depends(get_job_store),
):
    try:
        meta = store.get_meta(job_id)
        if meta is None:
            raise HTTPException(status_code=404, detail="Job not found. It may have expired.")
        result = AsyncResult(job_id, app=celery_app)
        return build_job_status(job_id, meta, result.state, result.result)
    except (RedisError, OperationalError):
        raise HTTPException(status_code=503, detail=QUEUE_UNAVAILABLE)
