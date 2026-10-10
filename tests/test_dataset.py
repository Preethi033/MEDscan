import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scanexplain.dataset import _ensure_dataset_file, load_pneumoniamnist_split


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


if __name__ == "__main__":
    unittest.main()
