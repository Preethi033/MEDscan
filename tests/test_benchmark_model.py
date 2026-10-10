import unittest

import numpy as np
from PIL import Image

from scanexplain.benchmark_model import PneumoniaMNISTBaseline
from scanexplain.dataset import DatasetSplit
from scanexplain.pipeline import ModelServices


def _synthetic_split():
    images = []
    labels = []
    for label in (0, 1):
        for index in range(4):
            image = np.zeros((224, 224), dtype=np.uint8)
            if label == 0:
                image[45 + index:150 + index, :] = 150
            else:
                image[:, 45 + index:150 + index] = 150
            image = np.clip(
                image.astype(np.int16) + index * 3,
                0,
                255,
            ).astype(np.uint8)
            images.append(image)
            labels.append(label)
    return DatasetSplit(np.stack(images), np.asarray(labels))


class BenchmarkModelTests(unittest.TestCase):
    def test_baseline_predicts_explains_and_screens_a_supported_image(self):
        split = _synthetic_split()
        model = PneumoniaMNISTBaseline(
            split_loader=lambda name: split,
        )
        services = ModelServices(
            predict_fn=model.predict,
            explain_fn=model.explain,
            embedding_fn=model.embed,
            ood_reference=lambda: model.reference,
            input_screen_fn=model.screen_input,
        )
        image = Image.fromarray(split.images[0]).convert("RGB")

        prediction = services.predict(image)
        overlay = services.explain(image)
        screening = services.screen(image, prediction)

        self.assertIn(prediction["label"], {"normal", "pneumonia"})
        self.assertAlmostEqual(sum(prediction["probs"].values()), 1.0)
        self.assertEqual(overlay.size, image.size)
        self.assertEqual(overlay.mode, "RGB")
        self.assertIsNotNone(screening.nearest_embedding_distance)
        self.assertIsNotNone(screening.embedding_distance_threshold)

    def test_screen_flags_blank_and_noise_images_as_suspected_ood(self):
        split = _synthetic_split()
        model = PneumoniaMNISTBaseline(
            split_loader=lambda name: split,
        )
        services = ModelServices(
            predict_fn=model.predict,
            embedding_fn=model.embed,
            ood_reference=lambda: model.reference,
            input_screen_fn=model.screen_input,
        )
        unsupported_images = (
            Image.new("RGB", (224, 224), color="black"),
            Image.fromarray(
                np.random.default_rng(4).integers(
                    0,
                    256,
                    (224, 224),
                    dtype=np.uint8,
                )
            ).convert("RGB"),
        )

        for image in unsupported_images:
            with self.subTest(image=image.getpixel((0, 0))):
                decision = services.screen(image, services.predict(image))
                self.assertEqual(decision.status, "ood_suspected")
                self.assertTrue(any("outside_supported_range" in reason for reason in decision.reasons))


if __name__ == "__main__":
    unittest.main()
