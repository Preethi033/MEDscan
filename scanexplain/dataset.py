"""Optional loader for the official PneumoniaMNIST educational benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any
from urllib.request import urlopen

DATASET_URL = (
    "https://zenodo.org/records/10519652/files/"
    "pneumoniamnist_224.npz?download=1"
)
DATASET_MD5 = "d6a3c71de1b945ea11211b03746c1fe1"
DATASET_FILENAME = "pneumoniamnist_224.npz"
LABEL_NAMES = {0: "normal", 1: "pneumonia"}
VALID_SPLITS = frozenset({"train", "val", "test"})
DATASET_ATTRIBUTION = (
    "PneumoniaMNIST (MedMNIST v3), CC BY 4.0. "
    "Dataset information: https://medmnist.com/ ; "
    "official distribution: https://doi.org/10.5281/zenodo.10519652"
)


@dataclass(frozen=True)
class DatasetSplit:
    images: Any
    labels: Any

    def __len__(self) -> int:
        return len(self.labels)


def load_pneumoniamnist_split(
    split: str = "train",
    *,
    data_dir: str | Path | None = None,
) -> DatasetSplit:
    """Download on first use, verify the official checksum, and load a split."""
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {', '.join(sorted(VALID_SPLITS))}")

    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "NumPy is required for dataset samples. Install with "
            "`python -m pip install '.[api]'`."
        ) from error

    root = Path(data_dir) if data_dir is not None else Path.home() / ".cache" / "scanexplain"
    dataset_path = _ensure_dataset_file(root)
    with np.load(dataset_path, allow_pickle=False) as archive:
        images = archive[f"{split}_images"]
        labels = archive[f"{split}_labels"].reshape(-1)
    return DatasetSplit(images=images, labels=labels)


def load_pneumoniamnist_gallery(
    *,
    split: str = "train",
    limit: int = 8,
    data_dir: str | Path | None = None,
) -> list[tuple[Any, str]]:
    """Return a small labeled image gallery from the requested dataset split."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    dataset = load_pneumoniamnist_split(split, data_dir=data_dir)

    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required to display dataset samples. Install the API "
            "dependencies with `python -m pip install '.[api]'`."
        ) from error

    selected_indices = []
    indices_by_label = {
        label_id: [index for index, value in enumerate(dataset.labels) if int(value) == label_id]
        for label_id in LABEL_NAMES
    }
    offset = 0
    while len(selected_indices) < limit:
        added = False
        for label_id in LABEL_NAMES:
            indices = indices_by_label[label_id]
            if offset < len(indices):
                selected_indices.append(indices[offset])
                added = True
                if len(selected_indices) == limit:
                    break
        if not added:
            break
        offset += 1

    examples = []
    for index in selected_indices:
        image = dataset.images[index]
        label = dataset.labels[index]
        label_id = int(label)
        examples.append(
            (
                Image.fromarray(image),
                f"Dataset label: {LABEL_NAMES.get(label_id, f'unknown ({label_id})')}",
            )
        )
    return examples


def make_pneumoniamnist_comparator(
    *,
    data_dir: str | Path | None = None,
):
    """Create a comparator that selects a labeled training example by class."""
    cache: DatasetSplit | None = None

    def compare(image: Any, prediction: dict[str, Any]) -> tuple[Any, str]:
        nonlocal cache
        if cache is None:
            cache = load_pneumoniamnist_split("train", data_dir=data_dir)
        label_name = prediction["label"]
        matching_label = next(
            (label_id for label_id, name in LABEL_NAMES.items() if name == label_name),
            None,
        )
        if matching_label is None:
            raise ValueError(f"Prediction label {label_name!r} is not a dataset class.")

        import numpy as np
        from PIL import Image

        matching_indices = np.flatnonzero(cache.labels == matching_label)
        if matching_indices.size == 0:
            raise RuntimeError(f"No training example found for label {label_name!r}.")
        example = Image.fromarray(cache.images[int(matching_indices[0])])
        return example, f"Training sample label: {label_name}"

    return compare


def _ensure_dataset_file(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    destination = root / DATASET_FILENAME
    if destination.exists():
        if _md5(destination) != DATASET_MD5:
            raise ValueError(
                f"Cached dataset checksum failed: {destination}. "
                "Move or remove that file before downloading again."
            )
        return destination

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{DATASET_FILENAME}.",
        suffix=".part",
        dir=root,
    )
    temporary_path = Path(temporary_name)
    try:
        digest = hashlib.md5()
        with os.fdopen(file_descriptor, "wb") as output:
            with urlopen(DATASET_URL, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
        if digest.hexdigest() != DATASET_MD5:
            raise ValueError(
                "Downloaded PneumoniaMNIST checksum did not match the official "
                "MedMNIST v3 checksum; the file was not installed."
            )
        temporary_path.replace(destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return destination


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
