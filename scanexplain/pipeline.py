"""Shared model interfaces and image preprocessing for the research demo."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
from typing import Any

from .trust import EmbeddingReference, OODDecision, check_ood

MAX_IMAGE_SIDE = 4096
MAX_IMAGE_PIXELS = 16_000_000


class FeatureNotConfigured(RuntimeError):
    """Raised when the app UI is called before a model component is connected."""


def preprocess_image(image: Any) -> Any:
    """Normalize a decoded image to RGB and reject excessive dimensions."""
    if image is None or not callable(getattr(image, "convert", None)):
        raise ValueError("Upload a supported image before running analysis.")

    width = getattr(image, "width", 0)
    height = getattr(image, "height", 0)
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("Image dimensions are invalid.")
    if width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE:
        raise ValueError(f"Image dimensions must not exceed {MAX_IMAGE_SIDE} pixels per side.")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"Image must not exceed {MAX_IMAGE_PIXELS:,} pixels.")

    return image.convert("RGB").copy()


class ModelServices:
    """Adapt person-specific model functions to the shared app contract.

    Each supplied function receives the same normalized RGB image. Functions
    may be omitted while the corresponding model work is in progress.
    """

    def __init__(
        self,
        *,
        predict_fn: Callable[[Any], Mapping[str, Any]] | None = None,
        explain_fn: Callable[[Any], Any] | None = None,
        segment_fn: Callable[[Any], Any] | None = None,
        embedding_fn: Callable[[Any], Any] | None = None,
        ood_reference: (
            EmbeddingReference | Callable[[], EmbeddingReference] | None
        ) = None,
        input_screen_fn: Callable[[Any], tuple[str, ...]] | None = None,
        compare_fn: Callable[[Any, Mapping[str, Any]], tuple[Any, str]] | None = None,
    ) -> None:
        self._predict_fn = predict_fn
        self._explain_fn = explain_fn
        self._segment_fn = segment_fn
        self._embedding_fn = embedding_fn
        self._ood_reference = ood_reference
        self._input_screen_fn = input_screen_fn
        self._compare_fn = compare_fn

    @property
    def feature_availability(self) -> dict[str, bool]:
        return {
            "prediction": self._predict_fn is not None,
            "explanation": self._explain_fn is not None,
            "segmentation": self._segment_fn is not None,
            "embedding_screening": (
                self._embedding_fn is not None and self._ood_reference is not None
            ),
            "input_quality_screening": self._input_screen_fn is not None,
            "comparison": self._compare_fn is not None,
        }

    def predict(self, image: Any) -> dict[str, Any]:
        return self.predict_preprocessed(preprocess_image(image))

    def predict_preprocessed(self, image: Any) -> dict[str, Any]:
        if self._predict_fn is None:
            raise FeatureNotConfigured("Classifier is not connected.")
        result = self._predict_fn(image)
        return _validate_prediction(result)

    def explain(self, image: Any) -> Any:
        return self.explain_preprocessed(preprocess_image(image))

    def explain_preprocessed(self, image: Any) -> Any:
        if self._explain_fn is None:
            raise FeatureNotConfigured("Explanation model is not connected.")
        return self._explain_fn(image)

    def segment(self, image: Any) -> Any:
        return self.segment_preprocessed(preprocess_image(image))

    def segment_preprocessed(self, image: Any) -> Any:
        if self._segment_fn is None:
            raise FeatureNotConfigured("Segmentation model is not connected.")
        return self._segment_fn(image)

    def screen(self, image: Any, prediction: Mapping[str, Any]) -> OODDecision:
        return self.screen_preprocessed(preprocess_image(image), prediction)

    def screen_preprocessed(
        self,
        image: Any,
        prediction: Mapping[str, Any],
    ) -> OODDecision:
        probabilities = tuple(prediction["probs"].values())
        embedding = None
        reference = self._ood_reference
        if reference is not None and self._embedding_fn is not None:
            embedding = self._embedding_fn(image)
            if callable(reference):
                reference = reference()
        decision = check_ood(probabilities, reference=reference, embedding=embedding)
        input_reasons = (
            self._input_screen_fn(image) if self._input_screen_fn is not None else ()
        )
        if not input_reasons:
            return decision
        return OODDecision(
            status="ood_suspected",
            confidence=decision.confidence,
            normalized_entropy=decision.normalized_entropy,
            nearest_embedding_distance=decision.nearest_embedding_distance,
            embedding_distance_threshold=decision.embedding_distance_threshold,
            reasons=tuple(dict.fromkeys((*decision.reasons, *input_reasons))),
        )

    def compare(self, image: Any, prediction: Mapping[str, Any]) -> tuple[Any, str]:
        return self.compare_preprocessed(preprocess_image(image), prediction)

    def compare_preprocessed(
        self,
        image: Any,
        prediction: Mapping[str, Any],
    ) -> tuple[Any, str]:
        if self._compare_fn is None:
            raise FeatureNotConfigured("Labeled-example comparison is not connected.")
        comparison_image, label = self._compare_fn(image, prediction)
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Comparison callback must return a non-empty example label.")
        return comparison_image, label


def _validate_prediction(result: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise ValueError("predict(image) must return a mapping with label and probs.")
    label = result.get("label")
    probs = result.get("probs")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("Prediction label must be a non-empty string.")
    if not isinstance(probs, Mapping) or not probs:
        raise ValueError("Prediction probs must be a non-empty class-to-probability mapping.")

    clean_probs: dict[str, float] = {}
    for class_name, probability in probs.items():
        if not isinstance(class_name, str) or not class_name:
            raise ValueError("Prediction probability keys must be non-empty class names.")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError(f"Probability for {class_name!r} must be numeric.")
        value = float(probability)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Probability for {class_name!r} must be between 0 and 1.")
        clean_probs[class_name] = value
    if not math.isclose(sum(clean_probs.values()), 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError("Prediction probabilities must sum to 1.")
    if label not in clean_probs:
        raise ValueError("Prediction label must be present in probs.")
    return {"label": label, "probs": clean_probs}
