import unittest

from scanexplain.pipeline import FeatureNotConfigured, ModelServices, preprocess_image


class FakeImage:
    def __init__(self, width=100, height=80, mode="L"):
        self.width = width
        self.height = height
        self.mode = mode

    def convert(self, mode):
        return FakeImage(self.width, self.height, mode)

    def copy(self):
        return FakeImage(self.width, self.height, self.mode)


class PipelineTests(unittest.TestCase):
    def test_preprocessing_normalizes_image_to_rgb_copy(self):
        result = preprocess_image(FakeImage())
        self.assertEqual(result.mode, "RGB")
        self.assertIsNot(result, None)

    def test_preprocessing_rejects_unsupported_or_oversized_images(self):
        with self.assertRaisesRegex(ValueError, "supported image"):
            preprocess_image(None)
        with self.assertRaisesRegex(ValueError, "per side"):
            preprocess_image(FakeImage(width=5000))
        with self.assertRaisesRegex(ValueError, "pixels"):
            preprocess_image(FakeImage(width=4001, height=4000))

    def test_model_functions_share_preprocessed_image_and_validate_output(self):
        seen_modes = []

        def predict(image):
            seen_modes.append(image.mode)
            return {"label": "class-a", "probs": {"class-a": 0.8, "class-b": 0.2}}

        services = ModelServices(predict_fn=predict)
        result = services.predict(FakeImage())

        self.assertEqual(result["label"], "class-a")
        self.assertEqual(result["probs"]["class-a"], 0.8)
        self.assertEqual(seen_modes, ["RGB"])

    def test_invalid_predictions_fail_explicitly(self):
        services = ModelServices(
            predict_fn=lambda image: {"label": "class-a", "probs": {"class-a": 0.7}}
        )
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            services.predict(FakeImage())

    def test_unconfigured_components_raise_explicit_status(self):
        with self.assertRaisesRegex(FeatureNotConfigured, "not connected"):
            ModelServices().predict(FakeImage())


if __name__ == "__main__":
    unittest.main()
