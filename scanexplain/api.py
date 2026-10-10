"""HTTP API used by the Expo Go research-demo client."""

from __future__ import annotations

import base64
from io import BytesIO
import os
from typing import Any

from .dataset import DATASET_ATTRIBUTION, load_pneumoniamnist_gallery
from .pipeline import FeatureNotConfigured, ModelServices, preprocess_image
from .trust import FailureReport

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
RESEARCH_NOTICE = (
    "Research prototype only. Not for diagnosis, treatment decisions, or "
    "emergency care. Predictions and visualizations are not clinical advice."
)


def create_app(
    services: ModelServices | None = None,
    *,
    failure_report: FailureReport | None = None,
) -> Any:
    try:
        from fastapi import FastAPI, File, HTTPException, Query, UploadFile
        from fastapi.middleware.cors import CORSMiddleware
        from PIL import Image, UnidentifiedImageError
    except ImportError as error:
        raise RuntimeError(
            "The API dependencies are required. Install with "
            "`python -m pip install '.[api]'`."
        ) from error

    backend = services or ModelServices()
    application = FastAPI(
        title="ScanExplain research API",
        description=RESEARCH_NOTICE,
        version="0.1.0",
    )
    cors_origins = [
        origin.strip()
        for origin in os.environ.get(
            "SCANEXPLAIN_CORS_ORIGINS",
            "http://localhost:8081,http://127.0.0.1:8081",
        ).split(",")
        if origin.strip()
    ]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @application.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "purpose": "research_demo",
            "research_notice": RESEARCH_NOTICE,
            "features": backend.feature_availability,
        }

    @application.post("/api/analyze")
    async def analyze(file: UploadFile = File(...)) -> dict[str, Any]:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds the 12 MB upload limit.")

        try:
            with Image.open(BytesIO(content)) as source:
                if source.format not in ALLOWED_IMAGE_FORMATS:
                    raise HTTPException(
                        status_code=415,
                        detail="Supported image formats are JPEG, PNG, and WebP.",
                    )
                normalized = preprocess_image(source)
        except HTTPException:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise HTTPException(status_code=400, detail=f"Invalid image: {error}") from error

        result: dict[str, Any] = {
            "prediction": None,
            "confidence": None,
            "screening": None,
            "explanation_overlay": None,
            "segmentation_overlay": None,
            "comparison": None,
            "features": backend.feature_availability,
            "research_notice": RESEARCH_NOTICE,
        }

        try:
            prediction = backend.predict_preprocessed(normalized)
        except FeatureNotConfigured:
            prediction = None
        if prediction is not None:
            result["prediction"] = prediction
            result["confidence"] = max(prediction["probs"].values())
            result["screening"] = backend.screen_preprocessed(normalized, prediction).to_dict()
            try:
                comparison_image, comparison_label = backend.compare_preprocessed(
                    normalized,
                    prediction,
                )
                result["comparison"] = {
                    "image": _image_data_url(comparison_image),
                    "label": comparison_label,
                }
            except FeatureNotConfigured:
                pass

        try:
            result["explanation_overlay"] = _image_data_url(
                backend.explain_preprocessed(normalized)
            )
        except FeatureNotConfigured:
            pass

        try:
            result["segmentation_overlay"] = _image_data_url(
                backend.segment_preprocessed(normalized)
            )
        except FeatureNotConfigured:
            pass

        return result

    @application.get("/api/failures")
    def failures() -> dict[str, Any]:
        if failure_report is None:
            return {
                "status": "not_connected",
                "message": "No labeled test-set failure report has been connected.",
                "failure_gallery": [],
            }
        return {"status": "available", **failure_report.to_dict()}

    @application.get("/api/samples")
    def samples(limit: int = Query(default=8, ge=1, le=12)) -> dict[str, Any]:
        examples = load_pneumoniamnist_gallery(split="train", limit=limit)
        return {
            "status": "available",
            "split": "train",
            "attribution": DATASET_ATTRIBUTION,
            "clinical_use": "not_intended",
            "examples": [
                {"image": _image_data_url(image), "label": label}
                for image, label in examples
            ],
        }

    return application


def _image_data_url(image: Any) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


app = create_app()
