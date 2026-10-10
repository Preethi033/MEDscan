import asyncio
from io import BytesIO
import unittest

try:
    from fastapi import HTTPException
    from PIL import Image
    from starlette.datastructures import Headers, UploadFile
    from scanexplain.api import create_app
    API_IMPORT_ERROR = None
except (ImportError, RuntimeError) as error:
    API_IMPORT_ERROR = error

from scanexplain.pipeline import ModelServices


def _upload(filename="sample.png", image_format="PNG"):
    content = BytesIO()
    Image.new("L", (32, 32), color=100).save(content, format=image_format)
    return UploadFile(
        file=BytesIO(content.getvalue()),
        filename=filename,
        headers=Headers({"content-type": f"image/{image_format.lower()}"}),
    )


@unittest.skipIf(API_IMPORT_ERROR is not None, "install the optional API dependencies")
class ApiHandlerTests(unittest.TestCase):
    @staticmethod
    def _routes(services=None):
        app = create_app(services)
        return {
            route.path: route.endpoint
            for route in app.routes
            if hasattr(route, "endpoint")
        }

    def test_health_and_stub_upload_report_no_fabricated_prediction(self):
        routes = self._routes()
        health = routes["/api/health"]()
        result = asyncio.run(routes["/api/analyze"](_upload()))

        self.assertEqual(health["status"], "ok")
        self.assertIsNone(result["prediction"])
        self.assertIsNone(result["screening"])
        self.assertFalse(result["features"]["prediction"])

    def test_configured_components_return_prediction_and_overlays(self):
        services = ModelServices(
            predict_fn=lambda image: {
                "label": "normal",
                "probs": {"normal": 0.8, "pneumonia": 0.2},
            },
            explain_fn=lambda image: image,
            segment_fn=lambda image: image,
        )
        result = asyncio.run(self._routes(services)["/api/analyze"](_upload()))

        self.assertEqual(result["prediction"]["label"], "normal")
        self.assertEqual(result["confidence"], 0.8)
        self.assertEqual(result["screening"]["status"], "cannot_assess")
        self.assertTrue(result["explanation_overlay"].startswith("data:image/png;base64,"))
        self.assertTrue(result["segmentation_overlay"].startswith("data:image/png;base64,"))

    def test_invalid_images_are_rejected(self):
        route = self._routes()["/api/analyze"]
        invalid = UploadFile(file=BytesIO(b"not an image"), filename="bad.png")

        with self.assertRaises(HTTPException) as raised:
            asyncio.run(route(invalid))

        self.assertEqual(raised.exception.status_code, 400)

    def test_cors_allows_local_expo_web_origin(self):
        application = create_app()
        cors = next(
            middleware
            for middleware in application.user_middleware
            if middleware.cls.__name__ == "CORSMiddleware"
        )

        self.assertIn("http://localhost:8081", cors.kwargs["allow_origins"])


if __name__ == "__main__":
    unittest.main()
