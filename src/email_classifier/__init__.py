from .classifier import DEFAULT_PROMPT, classify_email
from .models import (
    Category,
    ClassificationResult,
    ClassifierError,
    EmailClassification,
    PromptConfig,
)

__all__ = [
    "DEFAULT_PROMPT",
    "Category",
    "ClassificationResult",
    "ClassifierError",
    "EmailClassification",
    "PromptConfig",
    "classify_email",
]
