import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scanexplain.dataset import (
    _ensure_dataset_file,
    load_pneumoniamnist_sample,
    load_pneumoniamnist_split,
)


class FakeResponse:
    def __init__(self, content):
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self, size):
        content, self.content = self.content[:size], self.content[size:]
        return content


class DatasetTests(unittest.TestCase):
    def test_download_verifies_checksum_and_reuses_cached_file(self):
        content = b"verified benchmark archive"
        expected_md5 = hashlib.md5(content).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            with patch("scanexplain.dataset.DATASET_MD5", expected_md5):
                with patch(
                    "scanexplain.dataset.urlopen",
                    return_value=FakeResponse(content),
                ) as opener:
                    path = _ensure_dataset_file(Path(directory))
                    self.assertEqual(path.read_bytes(), content)
                    self.assertEqual(_ensure_dataset_file(Path(directory)), path)
                    opener.assert_called_once()

    def test_bad_download_is_rejected_and_partial_file_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("scanexplain.dataset.DATASET_MD5", "0" * 32):
                with patch(
                    "scanexplain.dataset.urlopen",
                    return_value=FakeResponse(b"unverified archive"),
                ):
                    with self.assertRaisesRegex(ValueError, "checksum did not match"):
                        _ensure_dataset_file(Path(directory))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_invalid_split_is_rejected_without_optional_dependencies(self):
        with self.assertRaisesRegex(ValueError, "split must be one of"):
            load_pneumoniamnist_split("validation")

    def test_sample_ids_accept_normal_and_pneumonia_classes(self):
        import numpy as np

        with patch(
            "scanexplain.dataset.load_pneumoniamnist_split",
            return_value=type(
                "Split",
                (),
                {
                    "images": np.asarray([[[1]], [[2]], [[3]], [[4]]], dtype=np.uint8),
                    "labels": np.asarray([0, 1, 0, 1]),
                },
            )(),
        ):
            from PIL import Image

            normal_image, normal_label = load_pneumoniamnist_sample(
                "PMN-TRAIN-NORMAL-0002"
            )
            pneumonia_image, pneumonia_label = load_pneumoniamnist_sample(
                "PMN-TRAIN-PNEUMONIA-0002"
            )

        self.assertIsInstance(normal_image, Image.Image)
        self.assertIsInstance(pneumonia_image, Image.Image)
        self.assertIn("normal", normal_label)
        self.assertIn("pneumonia", pneumonia_label)
        self.assertIn("not a diagnosis", normal_label)
        self.assertIn("not a diagnosis", pneumonia_label)


if __name__ == "__main__":
    unittest.main()
