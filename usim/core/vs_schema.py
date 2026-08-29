"""OpenAI ``response_format`` schemas for Verbalized Sampling.

Mirrors ``external/persuasion_simulation/src/llms/schema.py`` so the JSON we
get back is shape-guaranteed by the API rather than depending on the model
following a prompt-only instruction.

The ``random`` schema omits ``probability`` (matching the random VS prompt
which doesn't ask for it). The ``prob`` schema requires both ``text`` and
``probability``.
"""

from __future__ import annotations

from typing import Any, Dict


def _build_random_schema(num_samples: int) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "vs_random_responses",
            "schema": {
                "type": "object",
                "properties": {
                    "responses": {
                        "type": "array",
                        "description": (
                            f"Exactly {num_samples} plausible candidate "
                            "responses; each entry has a 'text' field."
                        ),
                        "minItems": num_samples,
                        "maxItems": num_samples,
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "The reply text.",
                                }
                            },
                            "required": ["text"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["responses"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


def _build_prob_schema(num_samples: int) -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "vs_prob_responses",
            "schema": {
                "type": "object",
                "properties": {
                    "responses": {
                        "type": "array",
                        "description": (
                            f"Exactly {num_samples} plausible candidate "
                            "responses with verbalized probabilities; each "
                            "entry has 'text' and 'probability' fields."
                        ),
                        "minItems": num_samples,
                        "maxItems": num_samples,
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "The reply text.",
                                },
                                "probability": {
                                    "type": "number",
                                    "description": (
                                        "How likely this reply is, in [0, 1]. "
                                        "Probabilities should roughly sum "
                                        "to 1.0."
                                    ),
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                            },
                            "required": ["text", "probability"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["responses"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


def get_vs_response_format(method: str, num_samples: int = 5) -> Dict[str, Any]:
    """Return the OpenAI ``response_format`` dict for a VS method.

    The returned schema pins the ``responses`` array to exactly
    ``num_samples`` items via ``minItems``/``maxItems``. This matters when
    the surrounding system prompt contains an explicit "one message at a
    time" instruction (e.g., tau2 user-simulator guidelines) that the
    model would otherwise treat as overriding the prompt-level VS request.

    Args:
        method: ``"prob"`` (text + probability) or ``"random"`` (text only).
        num_samples: Required length of the ``responses`` array.

    Raises:
        ValueError: if ``method`` is not one of the two supported modes,
            or if ``num_samples < 1``.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    if method == "random":
        return _build_random_schema(num_samples)
    if method == "prob":
        return _build_prob_schema(num_samples)
    raise ValueError(f"method must be 'prob' or 'random', got {method!r}")
