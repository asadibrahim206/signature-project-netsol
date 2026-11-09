from typing import Literal

from pydantic import BaseModel


class VerifyResponse(BaseModel):
    reference_filename: str | None
    candidate_filename: str | None
    score: float        # 0 to 1: how likely both signatures come from the same genuine hand
    threshold: float    # the score needed to count as a match (chosen when the model was trained)
    is_genuine: bool    # True when score >= threshold
    verdict: str        # a human-readable sentence for the user


class BoundingBox(BaseModel):
    """A rectangle around a signature, in pixels of the checked picture."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float


class Detection(BaseModel):
    confidence: float   # 0 to 1: how sure the model is that this box is a signature
    box: BoundingBox


class PageResult(BaseModel):
    page: int           # PDF page, or image number, counting from 1
    width: int
    height: int
    signature_found: bool
    detections: list[Detection]


class DetectResponse(BaseModel):
    filename: str | None
    file_type: str
    images_checked: int
    signature_detected: bool
    total_signatures: int
    confidence_threshold: float
    results: list[PageResult]
    verdict: str


# ---- Background jobs ----
class JobSubmitResponse(BaseModel):
    job_id: str
    status: Literal["queued"]
    status_url: str     # ask this address again and again until the job is done


class JobStatusResponse(BaseModel):
    job_id: str
    kind: str                                   # which kind of job, e.g. "detect"
    filename: str | None
    created_at: str
    status: Literal["queued", "running", "done", "failed"]
    result: DetectResponse | VerifyResponse | None = None   # only when status is "done"
    error: str | None = None                                # only when status is "failed"
