from pathlib import Path

from fastapi import HTTPException

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# extension -> (kind of file, the first bytes a real file of that type must have)
ALLOWED_TYPES = {
    ".png": ("image", b"\x89PNG\r\n\x1a\n"),
    ".jpg": ("image", b"\xff\xd8\xff"),
    ".jpeg": ("image", b"\xff\xd8\xff"),
    ".pdf": ("pdf", b"%PDF"),
    ".docx": ("docx", b"PK\x03\x04"),
}


def validate_upload(
    filename: str | None,
    contents: bytes,
    allowed_kinds: set[str] | None = None,
) -> str:
    """Check the uploaded file is allowed. Returns its kind: image, pdf or docx.

    allowed_kinds lets an endpoint accept only some kinds, e.g. {"image"}.
    If it is not given, every supported kind is accepted.
    """
    extension = Path(filename or "").suffix.lower()

    if extension not in ALLOWED_TYPES:
        allowed = ", ".join(ALLOWED_TYPES)
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Allowed types: {allowed}",
        )

    kind, expected_start = ALLOWED_TYPES[extension]

    if allowed_kinds is not None and kind not in allowed_kinds:
        accepted = ", ".join(sorted(allowed_kinds))
        raise HTTPException(
            status_code=415,
            detail=f"This endpoint only accepts: {accepted} files.",
        )

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File is too large (max 10 MB).")

    if not contents.startswith(expected_start):
        raise HTTPException(
            status_code=400,
            detail="File content does not match its extension.",
        )

    return kind
