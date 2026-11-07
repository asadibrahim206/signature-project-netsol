import io
import zipfile

import pymupdf  # reads and renders PDF pages
from fastapi import HTTPException
from PIL import Image

MAX_PDF_PAGES = 20
MAX_DOCX_IMAGES = 20
PDF_RENDER_DPI = 150
DOCX_IMAGE_FOLDER = "word/media/"
DOCX_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def extract_images(file_type: str, contents: bytes) -> list[Image.Image]:
    """Turn an uploaded file into a list of images the model can look at."""
    if file_type == "image":
        return [read_image(contents)]
    if file_type == "pdf":
        return _images_from_pdf(contents)
    if file_type == "docx":
        return _images_from_docx(contents)
    raise HTTPException(status_code=400, detail="Unsupported file kind.")


def read_image(data: bytes, convert_to_rgb: bool = True) -> Image.Image:
    """Decode image bytes into a Pillow image.

    convert_to_rgb=True gives a plain RGB picture (fine for most uses).
    convert_to_rgb=False keeps the original mode, e.g. a PNG with a transparent
    background, for code (like the verification model) that handles it itself.
    """
    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # forces Pillow to really decode it, so broken files fail here
        return image.convert("RGB") if convert_to_rgb else image
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the image file.")


def _images_from_pdf(data: bytes) -> list[Image.Image]:
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the PDF file.")

    with document:
        if document.needs_pass:
            raise HTTPException(status_code=400, detail="Password-protected PDFs are not supported.")
        if document.page_count > MAX_PDF_PAGES:
            raise HTTPException(
                status_code=400,
                detail=f"PDF has too many pages (max {MAX_PDF_PAGES}).",
            )

        images = []
        for page in document:
            pixmap = page.get_pixmap(dpi=PDF_RENDER_DPI)
            images.append(Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples))
        return images


def _images_from_docx(data: bytes) -> list[Image.Image]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Could not read the DOCX file.")

    with archive:
        names = archive.namelist()
        if "word/document.xml" not in names:
            raise HTTPException(status_code=400, detail="This is not a valid Word document.")

        image_names = [
            name
            for name in names
            if name.startswith(DOCX_IMAGE_FOLDER)
            and name.lower().endswith(DOCX_IMAGE_EXTENSIONS)
        ]
        if len(image_names) > MAX_DOCX_IMAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Document has too many images (max {MAX_DOCX_IMAGES}).",
            )

        return [read_image(archive.read(name)) for name in image_names]
