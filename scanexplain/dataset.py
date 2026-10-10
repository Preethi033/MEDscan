"""Optional loader for the official PneumoniaMNIST educational benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import re
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
    stat = dataset_path.stat()
    return _load_cached_split(
        str(dataset_path.resolve()),
        split,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        stat.st_size,
    )


@lru_cache(maxsize=6)
def _load_cached_split(
    path: str,
    split: str,
    _modified_ns: int,
    _changed_ns: int,
    _size: int,
) -> DatasetSplit:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "NumPy is required for dataset samples. Install with "
            "`python -m pip install '.[api]'`."
        ) from error

    with np.load(path, allow_pickle=False) as archive:
        images = archive[f"{split}_images"]
        labels = archive[f"{split}_labels"].reshape(-1)
    return DatasetSplit(images=images, labels=labels)


def load_pneumoniamnist_gallery(
    *,
    split: str = "train",
    limit: int = 8,
    label: str | None = None,
    data_dir: str | Path | None = None,
) -> list[tuple[Any, str]]:
    """Return labeled benchmark images, optionally filtered to one dataset class."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if label not in (None, "all", *LABEL_NAMES.values()):
        raise ValueError(
            f"label must be 'all' or one of {', '.join(LABEL_NAMES.values())}"
        )
    dataset = load_pneumoniamnist_split(split, data_dir=data_dir)

    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required to display dataset samples. Install the API "
            "dependencies with `python -m pip install '.[api]'`."
        ) from error

    if label in (None, "all"):
        selected_indices = []
        indices_by_label = {
            label_id: [
                index for index, value in enumerate(dataset.labels)
                if int(value) == label_id
            ]
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
    else:
        selected_label = next(
            label_id for label_id, name in LABEL_NAMES.items() if name == label
        )
        selected_indices = [
            index
            for index, value in enumerate(dataset.labels)
            if int(value) == selected_label
        ][:limit]

    examples = []
    for index in selected_indices:
        image = dataset.images[index]
        label = dataset.labels[index]
        label_id = int(label)
        examples.append(
            (
                Image.fromarray(image),
                f"Benchmark label: {LABEL_NAMES.get(label_id, f'unknown ({label_id})')} "
                "(not a diagnosis)",
            )
        )
    return examples


def load_pneumoniamnist_sample(
    sample_id: str,
    *,
    data_dir: str | Path | None = None,
) -> tuple[Any, str]:
    """Load a class-specific training sample by its generated gallery ID."""
    match = re.fullmatch(
        r"PMN-TRAIN-(?:(NORMAL|PNEUMONIA)-)?(\d{4})",
        sample_id,
    )
    if match is None:
        raise ValueError(
            "sample_id must be a generated PMN-TRAIN-#### or "
            "PMN-TRAIN-{NORMAL|PNEUMONIA}-#### sample ID"
        )
    class_name = (match.group(1) or "PNEUMONIA").lower()
    sample_number = int(match.group(2))
    if sample_number < 1:
        raise ValueError("sample_id index must be at least 1")

    dataset = load_pneumoniamnist_split("train", data_dir=data_dir)
    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "NumPy and Pillow are required to load a benchmark sample. "
            "Install the API dependencies with `python -m pip install '.[api]'`."
        ) from error

    label_id = next(
        label_id for label_id, name in LABEL_NAMES.items() if name == class_name
    )
    class_indices = np.flatnonzero(dataset.labels == label_id)
    if sample_number > len(class_indices):
        raise ValueError(f"Unknown {class_name} benchmark sample: {sample_id}")
    image_index = int(class_indices[sample_number - 1])
    image = Image.fromarray(dataset.images[image_index])
    label = f"Benchmark label: {class_name} (not a diagnosis)"
    return image, label


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
        stat = destination.stat()
        _verify_cached_dataset(
            str(destination.resolve()),
            DATASET_MD5,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            stat.st_size,
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


@lru_cache(maxsize=8)
def _verify_cached_dataset(
    path: str,
    expected_md5: str,
    _modified_ns: int,
    _changed_ns: int,
    _size: int,
) -> None:
    if _md5(Path(path)) != expected_md5:
        raise ValueError(
            f"Cached dataset checksum failed: {path}. "
            "Move or remove that file before downloading again."
        )
