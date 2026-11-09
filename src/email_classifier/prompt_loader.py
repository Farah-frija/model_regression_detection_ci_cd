import os
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .models import EmailClassification, PromptConfig

PROMPT_VERSION_ENV = "EMAIL_CLASSIFIER_PROMPT_VERSION"
DEFAULT_PROMPT_VERSION = "0.1.0"


class PromptExample(BaseModel):
    email_text: str
    expected_output: EmailClassification


class PromptFile(BaseModel):
    version: str
    timestamp: datetime
    system_prompt: str
    model: str = "gemini-3.5-flash-lite"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    examples: list[PromptExample] = Field(default_factory=list)


def load_prompt(path: str | Path | None = None) -> PromptConfig:
    load_dotenv()
    prompt_version = os.environ.get(PROMPT_VERSION_ENV, DEFAULT_PROMPT_VERSION)
    if path is not None:
        prompt_path = Path(path)
    else:
        if Path(prompt_version).name != prompt_version or prompt_version in {".", ".."}:
            raise ValueError(f"Invalid prompt version: {prompt_version!r}")
        prompt_path = (
            Path(__file__).resolve().parents[2] / "prompts" / f"{prompt_version}.yaml"
        )

    prompt_file = PromptFile.model_validate(
        yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    )
    if path is None and prompt_file.version != prompt_version:
        raise ValueError(
            f"Prompt file {prompt_path} declares version {prompt_file.version!r}, "
            f"but {PROMPT_VERSION_ENV} requested {prompt_version!r}"
        )

    system_prompt = prompt_file.system_prompt
    if prompt_file.examples:
        examples = "\n\n".join(
            f"Email: {example.email_text}\n"
            f"Expected JSON: {example.expected_output.model_dump_json()}"
            for example in prompt_file.examples
        )
        system_prompt = f"{system_prompt}\n\nFew-shot examples:\n{examples}"

    config_data = prompt_file.model_dump(exclude={"examples"})
    config_data["system_prompt"] = system_prompt
    return PromptConfig.model_validate(config_data)