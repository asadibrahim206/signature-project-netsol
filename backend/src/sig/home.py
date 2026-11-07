from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def home():
    return {
        "message": "Signature API is running",
        "endpoints": {
            "detection": "/detect",
            "verification": "/verify",
        },
    }
