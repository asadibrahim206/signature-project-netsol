from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, Request
from PIL import Image


@dataclass(frozen=True)
class VerificationResult:
    score: float
    threshold: float
    is_genuine: bool


class SignatureVerifier:
    """Wraps the trained verification model so the rest of the app never touches torch."""

    def __init__(self, model, config: dict, device: str = "cpu"):
        self._model = model
        self._config = config
        self._device = device

    @classmethod
    def load(cls, model_path: Path, device: str = "cpu") -> "SignatureVerifier":
        # Imported here and not at the top of the file: torch takes seconds to import,
        # so we only pay that cost when we really load the model.
        from signature_project.sigverify_core import load_checkpoint

        model, config = load_checkpoint(model_path, device)
        return cls(model, config, device)

    @property
    def threshold(self) -> float:
        return float(self._config["threshold"])

    def verify(self, reference: Image.Image, candidate: Image.Image) -> VerificationResult:
        from signature_project.sigverify_core import verify

        result = verify(self._model, self._config, [reference], candidate, self._device)
        return VerificationResult(
            score=result["score"],
            threshold=self.threshold,
            is_genuine=result["is_genuine"],
        )


def get_verifier(request: Request) -> SignatureVerifier:
    """FastAPI dependency: hands the loaded model to an endpoint, or answers 503."""
    verifier = getattr(request.app.state, "verifier", None)
    if verifier is None:
        raise HTTPException(
            status_code=503,
            detail="The verification model is not loaded.",
        )
    return verifier
