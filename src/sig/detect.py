from fastapi import APIRouter, File, UploadFile
from fastapi.concurrency import run_in_threadpool

from signature_project.extraction import extract_images
from signature_project.validation import validate_upload

router = APIRouter(tags=["Detection"])


@router.post("/detect")
async def detect_signature(file: UploadFile = File(...)):
    contents = await file.read()
    file_type = validate_upload(file.filename, contents)

    # Rendering PDFs is heavy work, so it runs in a separate thread
    # to keep the server responsive for other users.
    images = await run_in_threadpool(extract_images, file_type, contents)

    # The model is plugged in here in the next step.
    return {
        "filename": file.filename,
        "file_type": file_type,
        "size_bytes": len(contents),
        "images_extracted": len(images),
        "message": "File processed. Model prediction will be added in the next step.",
    }
