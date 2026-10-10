"""HTTP API used by the MedScan demo client."""

from __future__ import annotations

import base64
from io import BytesIO
import os
from typing import Any, Literal

from .benchmark_model import PneumoniaMNISTBaseline
from .dataset import (
    DATASET_ATTRIBUTION,
    load_pneumoniamnist_gallery,
    load_pneumoniamnist_sample,
)
from .pipeline import FeatureNotConfigured, ModelServices, preprocess_image
from .trust import FailureReport

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
DEMO_NOTICE = (
    "Educational demonstration only. Not for diagnosis, treatment decisions, or "
    "emergency care. Predictions and visualizations are not clinical advice."
)


def create_app(
    services: ModelServices | None = None,
    *,
    failure_report: FailureReport | None = None,
) -> Any:
    try:
        from fastapi import FastAPI, File, HTTPException, Query
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import Response
        from PIL import Image, UnidentifiedImageError
    except ImportError as error:
        raise RuntimeError(
            "The demo API dependencies are required. Install with "
            "`python -m pip install '.[api]'`."
        ) from error

    backend = services if services is not None else _default_model_services()
    application = FastAPI(
        title="MedScan demo API",
        description=DEMO_NOTICE,
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
            "purpose": "demo",
            "demo_notice": DEMO_NOTICE,
            "features": backend.feature_availability,
        }

    @application.post("/api/analyze")
    async def analyze(file: Any = File(...)) -> dict[str, Any]:
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

        return _analyze_normalized(normalized, backend)

    @application.post("/api/analyze-sample/{sample_id}")
    def analyze_sample(sample_id: str) -> dict[str, Any]:
        try:
            sample_image, dataset_label = load_pneumoniamnist_sample(sample_id)
            normalized = preprocess_image(sample_image)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        result = _analyze_normalized(normalized, backend)
        result["sample"] = {"sample_id": sample_id, "dataset_label": dataset_label}
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
    def samples(
        limit: int = Query(default=6, ge=1, le=6),
        label: Literal["normal", "pneumonia", "all"] = "all",
    ) -> dict[str, Any]:
        examples = load_pneumoniamnist_gallery(
            split="train",
            limit=limit,
            label=None if label == "all" else label,
        )
        class_counts = {"normal": 0, "pneumonia": 0}
        serialized_examples = []
        for image, benchmark_label in examples:
            class_name = next(
                name
                for name in class_counts
                if f"Benchmark label: {name}" in benchmark_label
            )
            class_counts[class_name] += 1
            sample_id = (
                f"PMN-TRAIN-{class_counts[class_name]:04d}"
                if label == "pneumonia"
                else f"PMN-TRAIN-{class_name.upper()}-{class_counts[class_name]:04d}"
            )
            serialized_examples.append(
                {
                    "sample_id": sample_id,
                    "image_url": f"/api/samples/{sample_id}/image",
                    "label": benchmark_label,
                }
            )
        return {
            "status": "available",
            "split": "train",
            "modality": "pediatric chest X-ray, 224x224 downsampled benchmark",
            "patient_records": False,
            "attribution": DATASET_ATTRIBUTION,
            "clinical_use": "not_intended",
            "examples": serialized_examples,
        }

    @application.get("/api/samples/{sample_id}/image")
    def sample_image(sample_id: str) -> Any:
        try:
            image, _ = load_pneumoniamnist_sample(sample_id)
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return Response(
            content=buffer.getvalue(),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    return application


def _image_data_url(image: Any) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _analyze_normalized(image: Any, backend: ModelServices) -> dict[str, Any]:
    result: dict[str, Any] = {
        "prediction": None,
        "confidence": None,
        "screening": None,
        "explanation_overlay": None,
        "segmentation_overlay": None,
        "comparison": None,
        "features": backend.feature_availability,
        "demo_notice": DEMO_NOTICE,
    }

    try:
        prediction = backend.predict_preprocessed(image)
    except FeatureNotConfigured:
        prediction = None
    if prediction is not None:
        result["prediction"] = prediction
        result["confidence"] = max(prediction["probs"].values())
        result["screening"] = backend.screen_preprocessed(image, prediction).to_dict()
        try:
            comparison_image, comparison_label = backend.compare_preprocessed(image, prediction)
            result["comparison"] = {
                "image": _image_data_url(comparison_image),
                "label": comparison_label,
            }
        except FeatureNotConfigured:
            pass

    try:
        result["explanation_overlay"] = _image_data_url(backend.explain_preprocessed(image))
    except FeatureNotConfigured:
        pass

    try:
        result["segmentation_overlay"] = _image_data_url(backend.segment_preprocessed(image))
    except FeatureNotConfigured:
        pass

    return result


def _default_model_services() -> ModelServices:
    model = PneumoniaMNISTBaseline()
    return ModelServices(
        predict_fn=model.predict,
        explain_fn=model.explain,
        embedding_fn=model.embed,
        ood_reference=lambda: model.reference,
        input_screen_fn=model.screen_input,
    )


app = create_app()
