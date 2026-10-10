"""Small, explicitly non-clinical PneumoniaMNIST baseline for the demo."""

from __future__ import annotations

from collections.abc import Callable
import threading
from typing import Any

from .dataset import DatasetSplit, LABEL_NAMES, load_pneumoniamnist_split
from .trust import EmbeddingReference

FEATURE_SIDE = 16
MAX_EPOCHS = 240
EARLY_STOPPING_PATIENCE = 30
LEARNING_RATE = 0.08
L2_STRENGTH = 0.01


class PneumoniaMNISTBaseline:
    """Train a small logistic classifier on benchmark training images only."""

    def __init__(
        self,
        *,
        data_dir: str | None = None,
        split_loader: Callable[..., DatasetSplit] = load_pneumoniamnist_split,
    ) -> None:
        self._data_dir = data_dir
        self._split_loader = split_loader
        self._lock = threading.Lock()
        self._mean: Any = None
        self._scale: Any = None
        self._weights: Any = None
        self._reference: EmbeddingReference | None = None
        self._profile_min: Any = None
        self._profile_max: Any = None

    def predict(self, image: Any) -> dict[str, Any]:
        import numpy as np

        self._fit_if_needed()
        row = self._standardize(_image_features(image))
        logit = float(np.clip(row @ self._weights[:-1] + self._weights[-1], -30, 30))
        pneumonia_score = 1.0 / (1.0 + np.exp(-logit))
        scores = {
            "normal": float(1.0 - pneumonia_score),
            "pneumonia": float(pneumonia_score),
        }
        label = max(scores, key=scores.get)
        return {"label": label, "probs": scores}

    def explain(self, image: Any) -> Any:
        """Return a simple linear-feature attribution overlay, not a lesion map."""
        import numpy as np
        from PIL import Image

        self._fit_if_needed()
        raw_features = _image_features(image)
        standardized = self._standardize(raw_features)
        contributions = np.abs(standardized * self._weights[:-1]).reshape(
            FEATURE_SIDE, FEATURE_SIDE
        )
        peak = float(contributions.max())
        if peak > 0:
            contributions = contributions / peak
        heat = Image.fromarray(
            np.uint8(np.clip(contributions * 255, 0, 255))
        ).resize(image.size, Image.Resampling.BILINEAR)
        heat_array = np.asarray(heat, dtype=np.float32) / 255.0
        red = np.full(heat_array.shape, 255, dtype=np.uint8)
        green = np.uint8(110 * (1.0 - heat_array))
        blue = np.zeros_like(heat_array, dtype=np.uint8)
        alpha = np.uint8(125 * heat_array)
        overlay = Image.fromarray(np.dstack((red, green, blue, alpha)))
        return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    def embed(self, image: Any) -> Any:
        """Return the standardized low-resolution image used by this baseline."""
        self._fit_if_needed()
        return self._standardize(_image_features(image))

    def screen_input(self, image: Any) -> tuple[str, ...]:
        """Flag extreme image profiles relative to the held-out validation split."""
        import numpy as np

        self._fit_if_needed()
        values = _image_profile(image)
        margins = np.asarray((0.04, 0.025, 0.004), dtype=np.float64)
        names = ("brightness", "contrast", "edge_activity")
        return tuple(
            f"image_{name}_outside_supported_range"
            for name, value, lower, upper in zip(
                names,
                values,
                self._profile_min - margins,
                self._profile_max + margins,
            )
            if value < lower or value > upper
        )

    @property
    def reference(self) -> EmbeddingReference:
        self._fit_if_needed()
        if self._reference is None:
            raise RuntimeError("The benchmark reference was not initialized.")
        return self._reference

    def _fit_if_needed(self) -> None:
        if self._weights is not None:
            return
        with self._lock:
            if self._weights is not None:
                return
            self._fit()

    def _fit(self) -> None:
        import numpy as np

        training = self._load_split("train")
        validation = self._load_split("val")
        train_features = _dataset_features(training.images)
        validation_features = _dataset_features(validation.images)
        labels = np.asarray(training.labels, dtype=np.float64)
        validation_labels = np.asarray(validation.labels, dtype=np.float64)
        if set(np.unique(labels)) != {0.0, 1.0}:
            raise ValueError("The PneumoniaMNIST training split must contain both classes.")
        if set(np.unique(validation_labels)) != {0.0, 1.0}:
            raise ValueError("The PneumoniaMNIST validation split must contain both classes.")

        mean = train_features.mean(axis=0)
        scale = train_features.std(axis=0)
        scale[scale < 1e-5] = 1.0
        train = (train_features - mean) / scale
        valid = (validation_features - mean) / scale
        train = np.column_stack((train, np.ones(len(train), dtype=np.float64)))
        valid = np.column_stack((valid, np.ones(len(valid), dtype=np.float64)))

        class_counts = np.bincount(labels.astype(np.int64), minlength=2)
        sample_weights = np.where(
            labels == 1,
            len(labels) / (2.0 * class_counts[1]),
            len(labels) / (2.0 * class_counts[0]),
        )
        weights = np.zeros(train.shape[1], dtype=np.float64)
        best_weights = weights.copy()
        best_loss = float("inf")
        stale_epochs = 0

        for _ in range(MAX_EPOCHS):
            logits = np.clip(train @ weights, -30, 30)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            gradient = train.T @ ((probabilities - labels) * sample_weights) / len(labels)
            gradient[:-1] += L2_STRENGTH * weights[:-1]
            weights -= LEARNING_RATE * gradient

            validation_logits = np.clip(valid @ weights, -30, 30)
            validation_probabilities = 1.0 / (1.0 + np.exp(-validation_logits))
            validation_loss = float(
                -np.mean(
                    validation_labels * np.log(validation_probabilities + 1e-9)
                    + (1.0 - validation_labels)
                    * np.log(1.0 - validation_probabilities + 1e-9)
                )
            )
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_weights = weights.copy()
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= EARLY_STOPPING_PATIENCE:
                    break

        reference_indices = np.linspace(
            0,
            len(valid) - 1,
            min(128, len(valid)),
            dtype=np.int64,
        )
        reference_vectors = valid[reference_indices, :-1]
        reference = EmbeddingReference.fit(reference_vectors.tolist())
        profiles = np.stack([_image_profile(image) for image in validation.images])
        profile_min, profile_max = np.quantile(profiles, (0.001, 0.999), axis=0)
        self._mean = mean
        self._scale = scale
        self._weights = best_weights
        self._reference = reference
        self._profile_min = profile_min
        self._profile_max = profile_max

    def _load_split(self, split: str) -> DatasetSplit:
        if self._data_dir is None:
            return self._split_loader(split)
        return self._split_loader(split, data_dir=self._data_dir)

    def _standardize(self, features: Any) -> Any:
        return (features - self._mean) / self._scale


def _image_features(image: Any) -> Any:
    import numpy as np
    from PIL import Image

    grayscale = image.convert("L").resize(
        (FEATURE_SIDE, FEATURE_SIDE),
        Image.Resampling.LANCZOS,
    )
    return np.asarray(grayscale, dtype=np.float64).reshape(-1) / 255.0


def _dataset_features(images: Any) -> Any:
    import numpy as np
    from PIL import Image

    return np.stack(
        [
            np.asarray(
                Image.fromarray(image).resize(
                    (FEATURE_SIDE, FEATURE_SIDE),
                    Image.Resampling.LANCZOS,
                ),
                dtype=np.float64,
            ).reshape(-1)
            / 255.0
            for image in images
        ]
    )


def _image_profile(image: Any) -> Any:
    import numpy as np
    from PIL import Image

    if not callable(getattr(image, "convert", None)):
        image = Image.fromarray(image)
    grayscale = image.convert("L").resize(
        (224, 224),
        Image.Resampling.LANCZOS,
    )
    values = np.asarray(grayscale, dtype=np.float64) / 255.0
    horizontal_activity = np.abs(np.diff(values, axis=1)).mean()
    vertical_activity = np.abs(np.diff(values, axis=0)).mean()
    return np.asarray(
        (
            values.mean(),
            values.std(),
            (horizontal_activity + vertical_activity) / 2.0,
        ),
        dtype=np.float64,
    )
