from pydantic import BaseModel


class VerifyResponse(BaseModel):
    reference_filename: str | None
    candidate_filename: str | None
    score: float        # 0 to 1: how likely both signatures come from the same genuine hand
    threshold: float    # the score needed to count as a match (chosen when the model was trained)
    is_genuine: bool    # True when score >= threshold
    verdict: str        # a human-readable sentence for the user
