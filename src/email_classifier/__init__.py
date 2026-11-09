from .classifier import DEFAULT_PROMPT, classify_email
from .models import (
    Category,
    ClassificationResult,
    ClassifierError,
    EmailClassification,
    PromptConfig,
)
from .prompt_loader import load_prompt

__all__ = [
    "DEFAULT_PROMPT",
    "Category",
    "ClassificationResult",
    "ClassifierError",
    "EmailClassification",
    "PromptConfig",
    "classify_email",
    "load_prompt",
]
