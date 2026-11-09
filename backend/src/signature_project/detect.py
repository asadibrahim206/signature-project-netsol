from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.concurrency import run_in_threadpool

from signature_project.detection_service import run_detection
from signature_project.detector import SignatureDetector, get_detector
from signature_project.schemas import DetectResponse

router = APIRouter(tags=["Detection"])


@router.post("/detect", response_model=DetectResponse)
async def detect_signature(
    file: UploadFile = File(...),
    detector: SignatureDetector = Depends(get_detector),
):
    contents = await file.read()
    # Rendering PDFs and running the model are heavy, so they run in a separate
    # thread to keep the server responsive for other users.
    return await run_in_threadpool(run_detection, detector, file.filename, contents)
