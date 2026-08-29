"""Tests for Verbalized Sampling structured-output schemas."""

import pytest

from usim.core.vs_schema import get_vs_response_format


def _responses_schema(method: str, num_samples: int = 5) -> dict:
    response_format = get_vs_response_format(method, num_samples)
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    return response_format["json_schema"]["schema"]["properties"]["responses"]


@pytest.mark.parametrize("method", ["prob", "random"])
def test_schema_requires_exact_sample_count(method: str) -> None:
    responses = _responses_schema(method, num_samples=3)
    assert responses["type"] == "array"
    assert responses["minItems"] == 3
    assert responses["maxItems"] == 3


def test_probability_schema_requires_bounded_probability() -> None:
    item = _responses_schema("prob")["items"]
    assert item["required"] == ["text", "probability"]
    assert item["additionalProperties"] is False
    assert item["properties"]["text"]["type"] == "string"
    assert item["properties"]["probability"] == {
        "type": "number",
        "description": (
            "How likely this reply is, in [0, 1]. "
            "Probabilities should roughly sum to 1.0."
        ),
        "minimum": 0.0,
        "maximum": 1.0,
    }


def test_random_schema_omits_probability() -> None:
    item = _responses_schema("random")["items"]
    assert item["required"] == ["text"]
    assert set(item["properties"]) == {"text"}
    assert item["additionalProperties"] is False


@pytest.mark.parametrize("num_samples", [0, -1])
def test_non_positive_sample_count_is_rejected(num_samples: int) -> None:
    with pytest.raises(ValueError, match="num_samples must be >= 1"):
        get_vs_response_format("prob", num_samples)


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="method must be 'prob' or 'random'"):
        get_vs_response_format("top_p", 5)
