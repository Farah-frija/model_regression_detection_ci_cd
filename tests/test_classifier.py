import json
from types import SimpleNamespace

import pytest
from google.genai import errors, types

from email_classifier import (
    DEFAULT_PROMPT,
    ClassifierError,
    EmailClassification,
    PromptConfig,
    classify_email,
)
from email_classifier.classifier import _load_api_key


def _response(text=None, block_reason=None, prompt_tokens=120, output_tokens=25, candidates=True):
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt_tokens, candidates_token_count=output_tokens
    )
    feedback = (
        types.GenerateContentResponsePromptFeedback(block_reason=block_reason)
        if block_reason
        else None
    )
    cands = []
    if candidates:
        content = types.Content(role="model", parts=[types.Part(text=text)] if text else [])
        cands = [types.Candidate(content=content, finish_reason=types.FinishReason.STOP)]
    return types.GenerateContentResponse(
        candidates=cands, usage_metadata=usage, prompt_feedback=feedback
    )


class FakeModels:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def _client(models: FakeModels):
    return SimpleNamespace(aio=SimpleNamespace(models=models))


def _json(category="billing", summary="Customer was charged twice this month."):
    return json.dumps({"category": category, "summary": summary})


async def test_returns_category_summary_and_measurements():
    models = FakeModels(response=_response(text=_json()))

    result = await classify_email("I was charged twice!", client=_client(models))

    assert result.category == "billing"
    assert result.summary == "Customer was charged twice this month."
    assert result.prompt_version == DEFAULT_PROMPT.version
    assert result.model == DEFAULT_PROMPT.model
    assert result.prompt_tokens == 120
    assert result.completion_tokens == 25
    assert result.total_tokens == 145
    assert result.latency_ms >= 0


async def test_prompt_config_is_passed_through_to_the_api():
    models = FakeModels(response=_response(text=_json(category="general", summary="x")))
    prompt = PromptConfig(
        version="9.9.9", system_prompt="custom", model="gemini-2.5-pro", temperature=0.5
    )

    result = await classify_email("hello", prompt=prompt, client=_client(models))

    call = models.calls[0]
    assert call["model"] == "gemini-2.5-pro"
    assert call["contents"] == "hello"
    assert call["config"].system_instruction == "custom"
    assert call["config"].temperature == 0.5
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].response_json_schema == EmailClassification.model_json_schema()
    assert result.prompt_version == "9.9.9"


async def test_blocked_prompt_raises_classifier_error():
    models = FakeModels(
        response=_response(block_reason=types.BlockedReason.SAFETY, candidates=False)
    )

    with pytest.raises(ClassifierError, match="blocked"):
        await classify_email("anything", client=_client(models))


async def test_empty_candidates_raise_classifier_error():
    models = FakeModels(response=_response(candidates=False))

    with pytest.raises(ClassifierError, match="no candidates"):
        await classify_email("anything", client=_client(models))


async def test_output_not_matching_schema_raises_classifier_error():
    models = FakeModels(response=_response(text=_json(category="sales")))

    with pytest.raises(ClassifierError, match="schema"):
        await classify_email("anything", client=_client(models))


async def test_non_json_output_raises_classifier_error():
    models = FakeModels(response=_response(text="not json at all"))

    with pytest.raises(ClassifierError, match="schema"):
        await classify_email("anything", client=_client(models))


async def test_api_error_is_wrapped_not_returned_as_a_result():
    error = errors.APIError(
        503, {"error": {"code": 503, "message": "service unavailable", "status": "UNAVAILABLE"}}
    )
    models = FakeModels(error=error)

    with pytest.raises(ClassifierError, match="service unavailable"):
        await classify_email("anything", client=_client(models))


async def test_missing_api_key_raises_classifier_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr("email_classifier.classifier.load_dotenv", lambda: None)

    with pytest.raises(ClassifierError, match="No Gemini API key"):
        await classify_email("anything")


def test_api_key_is_read_from_environment(monkeypatch):
    monkeypatch.setattr("email_classifier.classifier.load_dotenv", lambda: None)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    assert _load_api_key() == "test-key"


def test_google_api_key_is_used_as_fallback(monkeypatch):
    monkeypatch.setattr("email_classifier.classifier.load_dotenv", lambda: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "fallback-key")

    assert _load_api_key() == "fallback-key"


def test_category_must_be_one_of_the_four_allowed_values():
    with pytest.raises(ValueError):
        EmailClassification(category="sales", summary="not an allowed category")


def test_prompt_config_rejects_out_of_range_temperature():
    with pytest.raises(ValueError):
        PromptConfig(version="1", system_prompt="x", temperature=3.0)
