# Model Regression Detection System

CI-style regression detection for LLM-powered features. The goal is to catch
when a change to a model or prompt makes an LLM feature worse, before it ships.

The first feature under test is an **email classifier** for customer support.
It uses the **Gemini API** (`google-genai`).

## Current status

This is **Phase 1**: a single classifier function with tests. The later phases
are planned but not built yet.

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Classify one email with Gemini and return a typed result | Done |
| 2 | Load prompts from versioned YAML files | Planned |
| 3 | Eval engine: run a labelled email set through a prompt and report accuracy, latency and tokens | Planned |
| 4 | Regression gate: compare two prompt or model versions and fail CI on a drop | Planned |

The `PromptConfig` class has a `version` field so that Phase 2 and 3 can record
which prompt produced each result.

## What the classifier does

Given an email, `classify_email` sends it to Gemini with a system prompt and
returns a `ClassificationResult`:

- `category`: one of `billing`, `technical`, `account`, `general`
- `summary`: a one-sentence summary of the email
- `prompt_version`, `model`: which prompt and model produced the result
- `latency_ms`, `prompt_tokens`, `completion_tokens`: measurements for the eval engine

The reply is requested as JSON that matches the `EmailClassification` schema,
and the response is validated with Pydantic.

Errors are separated from wrong answers. Any failure raises `ClassifierError`,
so an outage or a blocked prompt is never counted as a wrong classification:

- the API call fails (wrapped from `google.genai.errors.APIError`)
- the prompt is blocked by Gemini's safety filters
- the model returns no candidates or no text
- the model's output does not match the schema

## Project layout

```
pyproject.toml                     dependencies and pytest config
.env.example                       template for the .env file with the API key
.gitignore                         keeps .env, .venv and caches out of git
src/email_classifier/
    models.py                      PromptConfig, EmailClassification,
                                   ClassificationResult, ClassifierError
    classifier.py                  DEFAULT_PROMPT and classify_email()
    __main__.py                    command-line entry point
    __init__.py                    public exports
tests/
    test_classifier.py             unit tests with a fake Gemini client
    test_main.py                   tests for the command-line entry point
```

## Setup

Requires Python 3.11 or newer.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

### Gemini API key

The key is read from a `.env` file in the project root. The file is listed in
`.gitignore`, so it is never committed.

1. Get a key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Copy the template and fill in the key:

   ```powershell
   copy .env.example .env
   ```

   ```
   # .env
   GEMINI_API_KEY=your-real-key
   ```

The classifier looks for `GEMINI_API_KEY`, then `GOOGLE_API_KEY`. A variable
already set in your shell takes priority over `.env`. If neither is set, it
raises `ClassifierError` with a message that points to `.env.example`.

Never commit `.env` or paste the key into code. `.env.example` is the only
env file that belongs in git.

## Usage

### Command line

The entry point is `email_classifier` run as a module. Pass the email as
arguments, or pipe it on stdin:

```powershell
.venv\Scripts\python.exe -m email_classifier "I was charged twice this month."

"I forgot my password." | .venv\Scripts\python.exe -m email_classifier
```

Example output:

```
Category:    billing
Summary:     The customer was charged twice this month.
Model:       gemini-3.5-flash-lite (prompt 0.1.0)
Latency:     1234 ms
Tokens:      111 (79 in, 32 out)
```

The exit code is `0` on success and `1` on an error, such as a missing API key
or a blocked prompt. Errors go to stderr.

### Python

```python
import asyncio
from email_classifier import classify_email

result = asyncio.run(classify_email("I was charged twice this month."))
print(result.category, result.summary, result.total_tokens)
```

The default model is `gemini-3.5-flash-lite`, set in `PromptConfig` in
`src/email_classifier/models.py`. To try another prompt or model, pass a
`PromptConfig`:

```python
from email_classifier import PromptConfig, classify_email

prompt = PromptConfig(
    version="0.2.0",
    system_prompt="...",
    model="gemini-2.5-pro",
    temperature=0.0,
)
result = await classify_email(email_text, prompt=prompt)
```

## Testing

```powershell
.venv\Scripts\python.exe -m pytest
```

The tests use a fake client that returns real `google.genai` response objects,
so they run offline and do not need an API key. They cover:

- the category, summary and token counts are read from the response
- the prompt config (model, system prompt, temperature, schema) reaches the API
- blocked prompts, empty responses, schema mismatches and API errors raise `ClassifierError`
- the category and temperature limits are enforced
- the command-line entry point prints the result, reads from stdin, and returns exit code 1 on errors
- a missing API key raises `ClassifierError`, and `GEMINI_API_KEY` is read from the environment

Nothing in the test suite calls the live Gemini API yet.


