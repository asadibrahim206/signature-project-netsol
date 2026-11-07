from fastapi import FastAPI

from signature_project import detect, home

app = FastAPI(
    title="Signature API",
    description="Signature detection and verification service",
    version="0.1.0",
)

app.include_router(home.router)
app.include_router(detect.router)
