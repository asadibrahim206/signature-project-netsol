from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image

from signature_project.extraction import extract_images
from signature_project.validation import validate_upload

router = APIRouter(tags=["Verification"])

# Verification compares one signature against another, so we only accept
# plain images here (a PDF with many pages would be ambiguous).
IMAGE_ONLY = {"image"}


async def _read_signature_image(upload: UploadFile, field_name: str) -> Image.Image:
    """Read, validate and decode one uploaded signature image."""
    contents = await upload.read()
    try:
        kind = validate_upload(upload.filename, contents, allowed_kinds=IMAGE_ONLY)
        images = await run_in_threadpool(extract_images, kind, contents)
    except HTTPException as error:
        # Say WHICH of the two uploads had the problem.
        raise HTTPException(
            status_code=error.status_code,
            detail=f"{field_name}: {error.detail}",
        )
    return images[0]


@router.post("/verify")
async def verify_signature(
    reference: UploadFile = File(..., description="The original, trusted signature"),
    candidate: UploadFile = File(..., description="The signature to check"),
):
    reference_image = await _read_signature_image(reference, "reference")
    candidate_image = await _read_signature_image(candidate, "candidate")

    # The verification model is plugged in here in a later step.
    return {
        "reference_filename": reference.filename,
        "candidate_filename": candidate.filename,
        "reference_size": list(reference_image.size),
        "candidate_size": list(candidate_image.size),
        "message": "Both images accepted. Model comparison will be added in a later step.",
    }
