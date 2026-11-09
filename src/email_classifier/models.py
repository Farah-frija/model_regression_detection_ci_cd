from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["billing", "technical", "account", "general"]


class PromptConfig(BaseModel):
    """The prompt under test. Phase 2 will load these from versioned YAML files."""

    version: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    system_prompt: str
    model: str = "gemini-3.5-flash-lite"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class EmailClassification(BaseModel):
    """Structured output the LLM must return. Its JSON schema is sent as the response schema."""

    category: Category
    summary: str = Field(description="One-sentence summary of the email.")


class ClassificationResult(BaseModel):
    """Classification plus the measurements the eval engine needs in Phase 3."""

    prompt_version: str
    model: str
    category: Category
    summary: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ClassifierError(Exception):
    """Raised when the API call fails or returns no usable output.

    Kept separate from wrong answers so an outage never shows up as a regression.
    """
