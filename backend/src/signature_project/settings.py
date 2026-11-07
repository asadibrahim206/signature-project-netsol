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
