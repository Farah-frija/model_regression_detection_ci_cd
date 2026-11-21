"""Backfill difficulty scores in the golden evaluation dataset.

Usage::

    .venv\\Scripts\\python.exe scripts/backfill_difficulty.py --dry-run
    .venv\\Scripts\\python.exe scripts/backfill_difficulty.py

The script reads ``data/eval/dataset_v0.1.0.json``, computes one difficulty
score per case from Gemini embeddings, and overwrites the existing
``expected_difficulty`` field (currently 0 for every case). It also adds
``is_edge_case`` (true when the embedding-inferred label differs from the
ground truth) and, for those cases, sets ``edge_case_type`` to
``"label_disagreement"``. It does not run the classifier under test, so the
scores are independent of the model and prompt being evaluated.

For every edge case the script records the cosine scores, difficulty score and
embedding model in the case's ``notes`` field. ``--dry-run`` performs the
embedding calls and prints summary diagnostics, but leaves the dataset
untouched.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# Allow running this file directly from a checkout that has not been installed
# with ``pip install -e .`` yet.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from email_classifier.difficulty import (  # noqa: E402
    EMBEDDING_MODEL,
    DifficultyResult,
    build_label_embeddings,
    compute_difficulty,
    create_client,
    embed_texts,
)
from email_classifier.models import ClassifierError  # noqa: E402

DEFAULT_DATASET = REPO_ROOT / "data" / "eval" / "dataset_v0.1.1.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Dataset JSON to update (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print scores without writing the file.",
    )
    return parser.parse_args(argv)


def load_dataset(path: Path) -> dict:
    """Read the dataset and check the one structural assumption we rely on."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError(f"{path} does not contain a 'cases' list.")
    if not payload["cases"]:
        raise ValueError(f"{path} contains no cases.")
    return payload


def save_dataset(path: Path, payload: dict) -> None:
    """Write the dataset back without reformatting its strings.

    ``ensure_ascii=False`` keeps the existing unicode characters as-is instead
    of turning them into ``\\u`` escapes.
    """
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def build_edge_case_notes(result: DifficultyResult) -> str:
    """Return the explanation stored in ``notes`` for a label disagreement."""
    return (
        f"Edge case: inferred label '{result.predicted_label}' disagrees with "
        f"true label '{result.true_label}'. Cosine scores between embeddings: "
        f"s_top={result.s_top:.4f} (email vs inferred label), "
        f"s_lab={result.s_lab:.4f} (inferred label vs true label). "
        f"difficulty_score={result.score:.4f}, "
        f"embedding_model={EMBEDDING_MODEL}."
    )


def main(argv: list[str] | None = None) -> int:
    """Embed every case and update difficulty and edge-case fields."""
    args = parse_args(argv)
    payload = load_dataset(args.dataset)
    cases = payload["cases"]

    client = create_client()
    label_embeddings = build_label_embeddings(client)
    email_embeddings = embed_texts(client, [case["input"] for case in cases])

    results: list[DifficultyResult] = []
    edge_cases: list[tuple[dict, DifficultyResult]] = []
    for case, email_embedding in zip(cases, email_embeddings):
        true_label = case["expected_output"]["category"]
        result = compute_difficulty(email_embedding, true_label, label_embeddings)
        results.append(result)

        is_edge_case = result.predicted_label != true_label
        if is_edge_case:
            edge_cases.append((case, result))
        if not args.dry_run:
            case["expected_difficulty"] = result.score
            case["is_edge_case"] = is_edge_case
            if is_edge_case:
                case["edge_case_type"] = "label_disagreement"
                case["notes"] = build_edge_case_notes(result)

    scores = [result.score for result in results]
    print(f"Scored {len(results)} cases in {args.dataset}")
    print(
        f"  min={min(scores):.4f}  mean={statistics.mean(scores):.4f}  "
        f"max={max(scores):.4f}"
    )

    print(
        "  embedding-inferred label disagrees with ground truth: "
        f"{len(edge_cases)} of {len(results)} cases (is_edge_case=true)"
    )
    hardest = sorted(zip(cases, results), key=lambda pair: pair[1].score, reverse=True)[:5]
    print("  hardest cases:")
    for case, result in hardest:
        print(
            f"    {case['id']}: score={result.score:.4f} "
            f"inferred={result.predicted_label} true={result.true_label} "
            f"s_top={result.s_top:.2f} s_lab={result.s_lab:.2f}"
        )

    if args.dry_run:
        print("Dry run: file not modified.")
    else:
        save_dataset(args.dataset, payload)
        print(f"Wrote {args.dataset}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ClassifierError, ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
