"""Pure-Python helpers for Verbalized Sampling JSON parsing.

This module intentionally has NO tau2 dependency so the parsing and sampling
logic can be unit-tested without installing tau2-bench. It is imported by
``vs_user_simulator.py`` (which does depend on tau2).
"""

from __future__ import annotations

import json
import random
import re
from typing import List, Optional


def strip_code_fence(text: str) -> str:
    r"""Strip ```json ... ``` markdown fences if present.

    Handles both ``\`\`\`json`` and bare ``\`\`\``. Non-fenced input passes
    through unchanged.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # drop the opening fence line
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def extract_json_object(text: str) -> Optional[dict]:
    """Best-effort JSON object extraction.

    Handles markdown fences and stray text around the JSON. Returns None on
    failure rather than raising so a transient parse error doesn't abort the
    episode.
    """
    if not text:
        return None
    text = strip_code_fence(text)
    # Fast path: whole string is already valid JSON
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Fallback: find the outermost {...} block and parse that
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return None
    try:
        return json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return None


def validate_candidates(obj: object) -> Optional[List[dict]]:
    """Pull a valid list of ``{text, probability}`` dicts out of a parsed JSON.

    Returns None if the JSON is the wrong shape or has no usable entries.
    Missing probabilities default to uniform over the list. Empty or
    non-string text fields are dropped.
    """
    if not isinstance(obj, dict):
        return None
    responses = obj.get("responses")
    if not isinstance(responses, list) or not responses:
        return None

    valid: List[dict] = []
    for r in responses:
        if not isinstance(r, dict):
            continue
        text = r.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        valid.append({
            "text": text.strip(),
            "probability": r.get("probability", 1.0 / len(responses)),
        })
    return valid if valid else None


def sample_candidate(candidates: List[dict], method: str = "prob") -> str:
    """Sample one candidate's 'text' from the parsed list.

    For ``method="prob"`` we weight by the verbalized probability (with a
    tiny floor so zero-probability entries aren't impossible). For
    ``method="random"`` we sample uniformly.
    """
    if not candidates:
        raise ValueError("Cannot sample from empty candidate list")
    if method not in ("prob", "random"):
        raise ValueError(f"method must be 'prob' or 'random', got {method!r}")

    if method == "random":
        return str(random.choice(candidates).get("text", ""))

    # Probability-weighted sampling
    weights: List[float] = []
    for c in candidates:
        prob = c.get("probability", 1.0 / len(candidates))
        try:
            prob = float(prob)
        except (TypeError, ValueError):
            prob = 1.0 / len(candidates)
        weights.append(max(prob, 1e-6))

    total = sum(weights)
    if total <= 0:
        return str(random.choice(candidates).get("text", ""))
    weights = [w / total for w in weights]
    chosen = random.choices(candidates, weights=weights, k=1)[0]
    return str(chosen.get("text", ""))
