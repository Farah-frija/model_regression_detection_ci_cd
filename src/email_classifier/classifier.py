import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from .models import ClassificationResult, ClassifierError, EmailClassification, PromptConfig

DEFAULT_PROMPT = PromptConfig(
    version="0.1.0",
    system_prompt=(
        "You classify customer support emails. Choose exactly one category: "
        "billing (payments, invoices, refunds, pricing), "
        "technical (bugs, errors, outages, how-to for the product), "
        "account (login, password, profile, access, deletion), "
        "or general (anything else). "
        "Then write a one-sentence summary of the email."
    ),
)


def _load_api_key() -> str:
    """Return the Gemini API key from the environment or a `.env` file.

    Variables already set in the environment win over `.env`.
    """
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ClassifierError(
            "No Gemini API key found. Set GEMINI_API_KEY in a .env file in the "
            "project root (copy .env.example to .env) or in the environment."
        )
    return api_key


async def classify_email(
    email_text: str,
    prompt: PromptConfig = DEFAULT_PROMPT,
    client: genai.Client | None = None,
) -> ClassificationResult:
    """Classify a customer email into a category and a one-sentence summary.

    Pass `client` to inject a fake in tests; otherwise a Gemini client is created
    with the key from GEMINI_API_KEY (or GOOGLE_API_KEY). Both are read from the
    environment, and from a `.env` file in the project root if one exists.
    """
    if client is None:
        client = genai.Client(api_key=_load_api_key())

    config = types.GenerateContentConfig(
        system_instruction=prompt.system_prompt,
        temperature=prompt.temperature,
        response_mime_type="application/json",
        response_json_schema=EmailClassification.model_json_schema(),
    )

    started = time.perf_counter()
    try:
        response = await client.aio.models.generate_content(
            model=prompt.model,
            contents=email_text,
            config=config,
        )
    except errors.APIError as exc:
        raise ClassifierError(f"Gemini API call failed: {exc}") from exc
    latency_ms = (time.perf_counter() - started) * 1000

    feedback = response.prompt_feedback
    if feedback is not None and feedback.block_reason:
        reason = feedback.block_reason_message or str(feedback.block_reason)
        raise ClassifierError(f"Prompt was blocked: {reason}")

    if not response.candidates:
        raise ClassifierError("Model returned no candidates")

    text = response.text
    if not text:
        finish = response.candidates[0].finish_reason
        raise ClassifierError(f"Model returned no output (finish reason: {finish})")

    try:
        parsed = EmailClassification.model_validate_json(text)
    except ValidationError as exc:
        raise ClassifierError(f"Model output did not match the schema: {exc}") from exc

    usage = response.usage_metadata
    return ClassificationResult(
        prompt_version=prompt.version,
        model=prompt.model,
        category=parsed.category,
        summary=parsed.summary,
        latency_ms=latency_ms,
        prompt_tokens=(usage.prompt_token_count or 0) if usage else 0,
        completion_tokens=(usage.candidates_token_count or 0) if usage else 0,
    )
