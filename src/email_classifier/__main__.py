"""Command-line entry point: `python -m email_classifier "email text"`.

The email is read from the command-line arguments, or from stdin if none are given.
Exit code is 0 on success and 1 on a ClassifierError (for example a missing API key).
"""

import asyncio
import sys

from .classifier import classify_email
from .models import ClassifierError


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    email_text = " ".join(args).strip() if args else sys.stdin.read().strip()

    if not email_text:
        print("No email text given. Pass it as an argument or pipe it on stdin.", file=sys.stderr)
        return 1

    try:
        result = asyncio.run(classify_email(email_text))
    except ClassifierError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Category:    {result.category}")
    print(f"Summary:     {result.summary}")
    print(f"Model:       {result.model} (prompt {result.prompt_version})")
    print(f"Latency:     {result.latency_ms:.0f} ms")
    print(f"Tokens:      {result.total_tokens} ({result.prompt_tokens} in, {result.completion_tokens} out)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
