"""Model-agnostic failure analysis and out-of-distribution review helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class PredictionRecord:
    """One labeled evaluation example and its model output.

    Shortcut cues are annotations supplied by an evaluator, not automatically
    inferred visual features.
    """

    sample_id: str
    true_label: str
    predicted_label: str
    confidence: float
    shortcut_cues: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class FailureCase:
    sample_id: str
    true_label: str
    predicted_label: str
    confidence: float
    reasons: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class FailureReport:
    evaluated_count: int
    misclassified_count: int
    low_confidence_count: int
    shortcut_cue_count: int
    failure_gallery: tuple[FailureCase, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report."""
        return asdict(self)


def analyze_predictions(
    records: Sequence[PredictionRecord],
    *,
    low_confidence_threshold: float = 0.6,
) -> FailureReport:
    """Summarize errors and review candidates from labeled model predictions."""
    if not 0.0 <= low_confidence_threshold <= 1.0:
        raise ValueError("low_confidence_threshold must be between 0 and 1")

    seen_ids: set[str] = set()
    cases: list[FailureCase] = []
    misclassified_count = 0
    low_confidence_count = 0
    shortcut_cue_count = 0

    for record in records:
        if not record.sample_id:
            raise ValueError("sample_id must not be empty")
        if record.sample_id in seen_ids:
            raise ValueError(f"duplicate sample_id: {record.sample_id}")
        seen_ids.add(record.sample_id)
        if not math.isfinite(record.confidence) or not 0.0 <= record.confidence <= 1.0:
            raise ValueError(f"confidence for {record.sample_id} must be between 0 and 1")

        reasons: list[str] = []
        if record.predicted_label != record.true_label:
            misclassified_count += 1
            reasons.append("misclassified")
        if record.confidence < low_confidence_threshold:
            low_confidence_count += 1
            reasons.append("low_confidence")
        if record.shortcut_cues:
            shortcut_cue_count += 1
            reasons.append("shortcut_cue")
        if not reasons:
            continue

        notes = []
        if "misclassified" in reasons:
            notes.append(
                f"Expected {record.true_label!r}, but predicted {record.predicted_label!r}."
            )
        if "low_confidence" in reasons:
            notes.append(f"Prediction confidence ({record.confidence:.3f}) is below "
                         f"{low_confidence_threshold:.3f}.")
        if record.shortcut_cues:
            cues = ", ".join(record.shortcut_cues)
            notes.append(
                f"Review annotated shortcut cue(s): {cues}; this annotation alone "
                "does not establish that the cue caused the prediction."
            )
        if record.note:
            notes.append(record.note)
        cases.append(
            FailureCase(
                sample_id=record.sample_id,
                true_label=record.true_label,
                predicted_label=record.predicted_label,
                confidence=record.confidence,
                reasons=tuple(reasons),
                note=" ".join(notes),
            )
        )

    return FailureReport(
        evaluated_count=len(records),
        misclassified_count=misclassified_count,
        low_confidence_count=low_confidence_count,
        shortcut_cue_count=shortcut_cue_count,
        failure_gallery=tuple(cases),
    )


@dataclass(frozen=True)
class EmbeddingReference:
    """Cosine-distance threshold calibrated from supported validation images."""

    embeddings: tuple[tuple[float, ...], ...]
    distance_threshold: float

    @classmethod
    def fit(
        cls,
        embeddings: Sequence[Sequence[float]],
        *,
        quantile: float = 0.95,
    ) -> EmbeddingReference:
        """Fit a nearest-neighbor threshold using leave-one-out distances.

        Use a representative, held-out set of known-supported images. The
        returned threshold is a screening heuristic, not a calibrated clinical
        or statistical guarantee.
        """
        if not 0.0 < quantile <= 1.0:
            raise ValueError("quantile must be greater than 0 and at most 1")
        vectors = tuple(_validated_vector(v, name="embedding") for v in embeddings)
        if len(vectors) < 2:
            raise ValueError("at least two supported reference embeddings are required")
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            raise ValueError("all reference embeddings must have the same dimension")

        loo_distances = []
        for index, vector in enumerate(vectors):
            loo_distances.append(
                min(
                    _cosine_distance(vector, other)
                    for other_index, other in enumerate(vectors)
                    if other_index != index
                )
            )
        return cls(vectors, _quantile(loo_distances, quantile))

    def nearest_distance(self, embedding: Sequence[float]) -> float:
        vector = _validated_vector(embedding, name="embedding")
        if len(vector) != len(self.embeddings[0]):
            raise ValueError("embedding dimension does not match reference embeddings")
        return min(_cosine_distance(vector, reference) for reference in self.embeddings)


@dataclass(frozen=True)
class OODDecision:
    status: str
    confidence: float
    normalized_entropy: float
    nearest_embedding_distance: float | None
    embedding_distance_threshold: float | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def check_ood(
    probabilities: Sequence[float],
    *,
    reference: EmbeddingReference | None = None,
    embedding: Sequence[float] | None = None,
    low_confidence_threshold: float = 0.5,
    high_entropy_threshold: float = 0.8,
) -> OODDecision:
    """Screen a prediction for uncertainty and embedding-space novelty.

    A high-confidence prediction is not proof that an upload is supported.
    Embedding-based screening is available only when a reference is supplied.
    """
    if not 0.0 <= low_confidence_threshold <= 1.0:
        raise ValueError("low_confidence_threshold must be between 0 and 1")
    if not 0.0 <= high_entropy_threshold <= 1.0:
        raise ValueError("high_entropy_threshold must be between 0 and 1")
    values = tuple(float(value) for value in probabilities)
    if not values:
        raise ValueError("probabilities must contain at least one class")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise ValueError("probabilities must be finite values between 0 and 1")
    if not math.isclose(sum(values), 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError("probabilities must sum to 1")
    if (reference is None) != (embedding is None):
        raise ValueError("reference and embedding must either both be supplied or both omitted")

    confidence = max(values)
    entropy = -sum(value * math.log(value) for value in values if value > 0.0)
    normalized_entropy = entropy / math.log(len(values)) if len(values) > 1 else 0.0
    distance = (
        reference.nearest_distance(embedding)
        if reference is not None and embedding is not None
        else None
    )

    reasons: list[str] = []
    if distance is not None and distance > reference.distance_threshold:
        reasons.append("embedding_far_from_supported_reference")
        status = "ood_suspected"
    else:
        if confidence < low_confidence_threshold:
            reasons.append("low_prediction_confidence")
        if normalized_entropy > high_entropy_threshold:
            reasons.append("high_prediction_entropy")
        if reasons:
            status = "uncertain"
        elif distance is None:
            status = "cannot_assess"
            reasons.append("no_supported_embedding_reference")
        else:
            status = "in_distribution_like"

    return OODDecision(
        status=status,
        confidence=confidence,
        normalized_entropy=normalized_entropy,
        nearest_embedding_distance=distance,
        embedding_distance_threshold=(
            reference.distance_threshold if reference is not None else None
        ),
        reasons=tuple(reasons),
    )


def _validated_vector(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if not vector or any(not math.isfinite(value) for value in vector):
        raise ValueError(f"{name} must be a non-empty vector of finite values")
    if all(value == 0.0 for value in vector):
        raise ValueError(f"{name} must not be a zero vector")
    return vector


def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimension does not match reference embeddings")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    similarity = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    return 1.0 - min(1.0, max(-1.0, similarity))


def _quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
