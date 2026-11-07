"""Tests that use the REAL trained model.

They are skipped automatically when torch/timm are not installed or when the model
file is not at models/sigverify.pt, so the rest of the suite still runs anywhere.
These tests check the plumbing (loading, score rules, symmetry, transparency).
They do NOT measure accuracy: for that you need real genuine and forged pairs.
"""
import random

import pytest
from PIL import Image, ImageDraw

pytest.importorskip("torch")
pytest.importorskip("timm")

from signature_project.settings import VERIFY_MODEL_PATH  # noqa: E402

pytestmark = pytest.mark.skipif(
    not VERIFY_MODEL_PATH.exists(),
    reason=f"model file not found at {VERIFY_MODEL_PATH}",
)


@pytest.fixture(scope="module")
def verifier():
    from signature_project.verifier import SignatureVerifier

    return SignatureVerifier.load(VERIFY_MODEL_PATH)


def draw_signature(seed: int, transparent: bool = False) -> Image.Image:
    """A made-up scribble that is always the same for the same seed."""
    rng = random.Random(seed)
    if transparent:
        image = Image.new("RGBA", (500, 220), (255, 255, 255, 0))
    else:
        image = Image.new("RGBA", (500, 220), (255, 255, 255, 255))
    points, x = [(40, 150)], 40
    while x < 460:
        x += rng.randint(15, 40)
        points.append((x, 110 + rng.randint(-70, 70)))
    ImageDraw.Draw(image).line(points, fill=(20, 20, 60, 255), width=4)
    return image


def test_threshold_comes_from_the_model_file(verifier):
    assert 0.0 < verifier.threshold < 1.0


def test_score_is_a_probability_and_follows_the_threshold_rule(verifier):
    result = verifier.verify(draw_signature(1), draw_signature(2))

    assert 0.0 <= result.score <= 1.0
    assert result.is_genuine == (result.score >= result.threshold)


def test_the_same_signature_is_accepted(verifier):
    signature = draw_signature(1)

    result = verifier.verify(signature, signature)

    assert result.is_genuine is True


def test_swapping_the_two_signatures_gives_the_same_score(verifier):
    a, b = draw_signature(1), draw_signature(2)

    forward = verifier.verify(a, b).score
    backward = verifier.verify(b, a).score

    assert abs(forward - backward) < 1e-4


def test_transparent_background_scores_like_a_white_background(verifier):
    white = draw_signature(1)
    transparent = draw_signature(1, transparent=True)

    score_white = verifier.verify(white, white).score
    score_transparent = verifier.verify(white, transparent).score

    assert abs(score_white - score_transparent) < 1e-3
