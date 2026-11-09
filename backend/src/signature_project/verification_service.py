from fastapi import HTTPException
from PIL import Image

from signature_project.extraction import read_image
from signature_project.schemas import VerifyResponse
from signature_project.validation import validate_upload
from signature_project.verifier import SignatureVerifier, VerificationResult

# Verification compares one signature against another, so we only accept
# plain images (a PDF with many pages would be ambiguous).
IMAGE_ONLY = {"image"}


def read_signature_image(filename: str | None, contents: bytes, field_name: str) -> Image.Image:
    """Validate and decode one signature image.

    Raises HTTPException that says WHICH upload ("reference" or "candidate") had the problem.
    """
    try:
        validate_upload(filename, contents, allowed_kinds=IMAGE_ONLY)
        # Keep the original mode (e.g. a transparent PNG): the model's own
        # preprocessing handles it, exactly as it did during training.
        return read_image(contents, False)
    except HTTPException as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=f"{field_name}: {error.detail}",
        )


def build_verify_response(
    reference_filename: str | None,
    candidate_filename: str | None,
    result: VerificationResult,
) -> VerifyResponse:
    """Turn the model's answer into the API answer (used by /verify and the worker)."""
    if result.is_genuine:
        verdict = "The signature matches the reference."
    else:
        verdict = "The signature does not match the reference."

    return VerifyResponse(
        reference_filename=reference_filename,
        candidate_filename=candidate_filename,
        # 6 decimals: this model's scores are squeezed close to 1.0, and the decision
        # threshold is 0.99933, so 4 decimals would hide the difference that matters.
        score=round(result.score, 6),
        threshold=round(result.threshold, 6),
        is_genuine=result.is_genuine,
        verdict=verdict,
    )


def run_verification(
    verifier: SignatureVerifier,
    reference_filename: str | None,
    reference_contents: bytes,
    candidate_filename: str | None,
    candidate_contents: bytes,
) -> VerifyResponse:
    """The whole verification job, used by the background worker."""
    reference = read_signature_image(reference_filename, reference_contents, "reference")
    candidate = read_signature_image(candidate_filename, candidate_contents, "candidate")
    result = verifier.verify(reference, candidate)
    return build_verify_response(reference_filename, candidate_filename, result)
