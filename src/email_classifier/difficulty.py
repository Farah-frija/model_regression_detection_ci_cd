"""Compute a numeric difficulty score for a labelled email.

The score is a proxy for how hard a case is likely to be for an email
classifier. It is derived only from text embeddings: the classifier under test
is never called. That matters for regression detection, because the score must
not change when the model or prompt being evaluated changes.

Method
------
1. Embed the raw email with an embedding model (Gemini in this project).
2. Embed a fixed description of each of the four categories with the *same*
   encoder.
3. Compare the email vector to every category vector with cosine similarity.
   The closest category is the "inferred" label; ``s_top`` is its score and
   ``margin`` is the gap to the runner-up.
4. Compare the inferred category vector to the ground-truth category vector.
   That similarity is ``s_lab`` (1 when they are the same category).
5. Combine the inverted signals into a score in [0, 1]: low ``s_top``, low
   ``s_lab`` and a small ``margin`` all make a case harder.

Only the raw email text is embedded; the dataset's ground-truth summary is
deliberately not used, because it was written to match the answer key.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Mapping, Sequence

from dotenv import load_dotenv
from google import genai

from .models import Category, ClassifierError

# Fixed category definitions. They are intentionally independent of the
# versioned prompts: a difficulty score must stay stable when a prompt changes.
LABELS: tuple[Category, ...] = ("billing", "technical", "account", "general")

LABEL_DESCRIPTIONS: dict[Category, str] = {
    "billing": "Questions about payments, invoices, refunds, charges, and pricing.",
    "technical": "Questions about product bugs, errors, outages, and how to use the product.",
    "account": "Questions about login, passwords, profiles, access rights, and account deletion.",
    "general": "Other topics that do not fit billing, technical, or account.",
}

# Gemini's text embedding model. Kept in one place so scores are reproducible.
EMBEDDING_MODEL = "gemini-embedding-001"

# Relative contribution of each signal. These must add up to 1 so that the
# final score stays in [0, 1]. Agreement with the ground truth matters most.
WEIGHT_S_TOP = 0.25
WEIGHT_S_LAB = 0.50
WEIGHT_MARGIN = 0.25

# Scores are stored in the dataset with four decimal places.
SCORE_DECIMALS = 4


@dataclass(frozen=True)
class DifficultyResult:
    """A score together with the signals that produced it.

    The extra fields are not written to the dataset; the backfill script prints
    them so a few cases can be spot-checked after a run.
    """

    score: float
    predicted_label: Category
    true_label: Category
    s_top: float
    s_lab: float
    margin: float


def load_api_key() -> str:
    """Return the Gemini API key from the environment or the project ``.env``.

    Shell variables take priority over ``.env``, matching the classifier.
    """
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ClassifierError(
            "No Gemini API key found. Set GEMINI_API_KEY in a .env file in the "
            "project root (copy .env.example to .env) or in the environment."
        )
    return api_key


def create_client() -> genai.Client:
    """Create the Gemini client used for embedding calls."""
    return genai.Client(api_key=load_api_key())


def _normalize(vector: Sequence[float]) -> list[float]:
    """Return a unit-length copy of ``vector``.

    Cosine similarity only needs direction, but normalizing before averaging
    stops whichever text embeds with the larger magnitude from dominating.
    """
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-length embedding.")
    return [component / norm for component in vector]


def _clamp_unit(value: float) -> float:
    """Keep a penalty in [0, 1] even if a cosine is slightly negative."""
    return max(0.0, min(1.0, value))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the cosine similarity of two vectors (1 means same direction)."""
    if len(a) != len(b):
        raise ValueError("Embeddings must have the same dimensionality.")

    norm_a = math.sqrt(sum(component * component for component in a))
    norm_b = math.sqrt(sum(component * component for component in b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Cosine similarity is undefined for zero-length vectors.")

    return sum(x * y for x, y in zip(a, b)) / (norm_a * norm_b)


def embed_texts(client: genai.Client, texts: Sequence[str]) -> list[list[float]]:
    """Embed all ``texts`` in one batched call and L2-normalize the results.

    Batching keeps a 50-case dataset to a single request. The returned vectors
    are safe to average or compare with ``cosine_similarity``.
    """
    texts = list(texts)
    if not texts:
        return []

    response = client.models.embed_content(model=EMBEDDING_MODEL, contents=texts)
    embeddings = response.embeddings or []
    if len(embeddings) != len(texts):
        raise ClassifierError(
            f"Embedding model returned {len(embeddings)} vectors for "
            f"{len(texts)} texts."
        )

    vectors: list[list[float]] = []
    for embedding in embeddings:
        if not embedding.values:
            raise ClassifierError("Embedding model returned an empty vector.")
        vectors.append(_normalize(embedding.values))
    return vectors


def build_label_embeddings(client: genai.Client) -> dict[Category, list[float]]:
    """Embed the four fixed category descriptions with the email encoder."""
    vectors = embed_texts(client, [LABEL_DESCRIPTIONS[label] for label in LABELS])
    return dict(zip(LABELS, vectors))


def compute_difficulty(
    email_embedding: Sequence[float],
    true_label: Category,
    label_embeddings: Mapping[Category, Sequence[float]],
) -> DifficultyResult:
    """Score one case. A higher score means a harder case.

    ``s_top`` is the similarity between the email and its closest category,
    ``s_lab`` is the similarity between that category and the ground truth,
    and ``margin`` is the gap between the best and second-best categories.

    Each signal is inverted so that harder cases score higher:
      * low ``s_top`` -> no category fits the email well;
      * low ``s_lab`` -> the obvious category is not the correct one;
      * small ``margin`` -> the case sits on a category boundary.
    """
    if true_label not in label_embeddings:
        raise ValueError(f"Unknown ground-truth label: {true_label!r}")

    similarities = {
        label: cosine_similarity(email_embedding, vector)
        for label, vector in label_embeddings.items()
    }
    ranked = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    predicted_label, s_top = ranked[0]
    s_second = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = s_top - s_second

    s_lab = cosine_similarity(
        label_embeddings[predicted_label], label_embeddings[true_label]
    )

    score = (
        WEIGHT_S_TOP * _clamp_unit(1.0 - s_top)
        + WEIGHT_S_LAB * _clamp_unit(1.0 - s_lab)
        + WEIGHT_MARGIN * _clamp_unit(1.0 - margin)
    )

    return DifficultyResult(
        score=round(score, SCORE_DECIMALS),
        predicted_label=predicted_label,
        true_label=true_label,
        s_top=round(s_top, SCORE_DECIMALS),
        s_lab=round(s_lab, SCORE_DECIMALS),
        margin=round(margin, SCORE_DECIMALS),
    )
