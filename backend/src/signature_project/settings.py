import os
from pathlib import Path

# The backend folder (the one that holds pyproject.toml).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Where the trained verification model lives. Override with the SIGVERIFY_MODEL_PATH
# environment variable if you keep it somewhere else (for example on a server).
VERIFY_MODEL_PATH = Path(
    os.getenv("SIGVERIFY_MODEL_PATH", PROJECT_ROOT / "models" / "sigverify.pt")
)

# "cpu" works everywhere. Use "cuda" on a machine with an NVIDIA GPU.
MODEL_DEVICE = os.getenv("SIGVERIFY_DEVICE", "cpu")

# ---- Detection model (exported from the Kaggle training notebook as ONNX) ----
DETECT_MODEL_PATH = Path(
    os.getenv("SIGDETECT_MODEL_PATH", PROJECT_ROOT / "models" / "signature_detector.onnx")
)

# A box needs at least this confidence to count as a signature (0 to 1).
# Higher = fewer false alarms but more missed signatures. We tune it after testing.
DETECT_CONF_THRESHOLD = float(os.getenv("SIGDETECT_CONF", "0.5"))

# Two boxes overlapping more than this are treated as the same signature.
DETECT_NMS_THRESHOLD = float(os.getenv("SIGDETECT_NMS", "0.45"))

# ---- Job queue (Redis + Celery) ----
# Redis is both the waiting line (the "broker") and the place results are kept.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# How long (seconds) uploads, job records and results stay in Redis before they expire.
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "3600"))

# A job that runs longer than this (seconds) is stopped (not enforced by the Windows "solo" pool).
JOB_TIME_LIMIT_SECONDS = int(os.getenv("JOB_TIME_LIMIT_SECONDS", "120"))
