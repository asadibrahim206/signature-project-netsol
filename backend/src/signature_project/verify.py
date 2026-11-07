from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image

from signature_project.extraction import read_image
from signature_project.schemas import VerifyResponse
from signature_project.validation import validate_upload
from signature_project.verifier import SignatureVerifier, get_verifier

router = APIRouter(tags=["Verification"])

# Verification compares one signature against another, so we only accept
# plain images here (a PDF with many pages would be ambiguous).
IMAGE_ONLY = {"image"}


async def _read_signature_image(upload: UploadFile, field_name: str) -> Image.Image:
    """Read, validate and decode one uploaded signature image."""
    contents = await upload.read()
    try:
        validate_upload(upload.filename, contents, allowed_kinds=IMAGE_ONLY)
        # Keep the original mode (for example a transparent PNG): the model's own
        # preprocessing knows how to handle it, exactly as it did during training.
        return await run_in_threadpool(read_image, contents, False)
    except HTTPException as error:
        # Say WHICH of the two uploads had the problem.
        raise HTTPException(
            status_code=error.status_code,
            detail=f"{field_name}: {error.detail}",
        )


@router.post("/verify", response_model=VerifyResponse)
async def verify_signature(
    reference: UploadFile = File(..., description="The original, trusted signature"),
    candidate: UploadFile = File(..., description="The signature to check"),
    verifier: SignatureVerifier = Depends(get_verifier),
):
    reference_image = await _read_signature_image(reference, "reference")
    candidate_image = await _read_signature_image(candidate, "candidate")

    # Running the model is heavy work, so it goes to a separate thread.
    result = await run_in_threadpool(verifier.verify, reference_image, candidate_image)

    if result.is_genuine:
        verdict = "The signature matches the reference."
    else:
        verdict = "The signature does not match the reference."

    return VerifyResponse(
        reference_filename=reference.filename,
        candidate_filename=candidate.filename,
        score=round(result.score, 4),
        threshold=round(result.threshold, 4),
        is_genuine=result.is_genuine,
        verdict=verdict,
    )
