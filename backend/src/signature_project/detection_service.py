from PIL import Image

from signature_project.detector import SignatureDetector
from signature_project.extraction import extract_images
from signature_project.schemas import BoundingBox, DetectResponse, Detection, PageResult
from signature_project.validation import validate_upload


def check_all_images(detector: SignatureDetector, images: list[Image.Image]) -> list[PageResult]:
    """Run the model on every picture (PDF page / embedded image) of the upload."""
    results = []
    for number, image in enumerate(images, start=1):
        boxes = detector.detect(image)
        results.append(
            PageResult(
                page=number,
                width=image.width,
                height=image.height,
                signature_found=bool(boxes),
                detections=[
                    Detection(
                        confidence=round(box.confidence, 4),
                        box=BoundingBox(
                            x_min=round(box.x_min, 1),
                            y_min=round(box.y_min, 1),
                            x_max=round(box.x_max, 1),
                            y_max=round(box.y_max, 1),
                        ),
                    )
                    for box in boxes
                ],
            )
        )
    return results


def build_verdict(images_checked: int, total_signatures: int) -> str:
    if images_checked == 0:
        return "No images were found in this document, so no signature could be checked."
    if total_signatures == 0:
        return "No signature detected."
    if total_signatures == 1:
        return "Signature detected (1 found)."
    return f"Signature detected ({total_signatures} found)."


def run_detection(detector: SignatureDetector, filename: str | None, contents: bytes) -> DetectResponse:
    """The whole detection job: check the file, turn it into pictures, run the model.

    Both the direct /detect endpoint and the background worker call this one function,
    so they can never disagree. Raises HTTPException for a bad file.
    """
    file_type = validate_upload(filename, contents)
    images = extract_images(file_type, contents)
    results = check_all_images(detector, images)

    total_signatures = sum(len(page.detections) for page in results)
    return DetectResponse(
        filename=filename,
        file_type=file_type,
        images_checked=len(images),
        signature_detected=total_signatures > 0,
        total_signatures=total_signatures,
        confidence_threshold=detector.conf_threshold,
        results=results,
        verdict=build_verdict(len(images), total_signatures),
    )
