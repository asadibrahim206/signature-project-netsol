import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from signature_project import detect, home, jobs, verify
from signature_project.detector import SignatureDetector
from signature_project.settings import (
    DETECT_CONF_THRESHOLD,
    DETECT_MODEL_PATH,
    DETECT_NMS_THRESHOLD,
    MODEL_DEVICE,
    VERIFY_MODEL_PATH,
)
from signature_project.verifier import SignatureVerifier

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts: load each model a single time and keep it
    # in memory, so requests don't pay the loading cost.
    app.state.verifier = None
    app.state.detector = None

    if VERIFY_MODEL_PATH.exists():
        app.state.verifier = SignatureVerifier.load(VERIFY_MODEL_PATH, MODEL_DEVICE)
        logger.info("Verification model loaded from %s", VERIFY_MODEL_PATH)
    else:
        logger.warning(
            "Verification model not found at %s - /verify will answer 503.",
            VERIFY_MODEL_PATH,
        )

    if DETECT_MODEL_PATH.exists():
        app.state.detector = SignatureDetector.load(
            DETECT_MODEL_PATH, DETECT_CONF_THRESHOLD, DETECT_NMS_THRESHOLD
        )
        logger.info("Detection model loaded from %s", DETECT_MODEL_PATH)
    else:
        logger.warning(
            "Detection model not found at %s - /detect will answer 503.",
            DETECT_MODEL_PATH,
        )

    yield
    app.state.verifier = None
    app.state.detector = None


app = FastAPI(
    title="Signature API",
    description="Signature detection and verification service",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(home.router)
app.include_router(detect.router)
app.include_router(verify.router)
app.include_router(jobs.router)
