"""
FastAPI routes — the complete inference pipeline.

POST /predict       — Main MRI analysis endpoint
GET  /health        — Service health check
GET  /model-info    — Model metadata
"""
from typing import Any
import cv2
import numpy as np
import tensorflow as tf
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.api.schemas import (
    PredictResponse, AnalysisResult, SegmentationResult,
    VisualizationsResult, ModelInfo, HealthResponse, ModelInfoResponse,
    ErrorResponse, ErrorDetail,
)
from app.core.config import settings
from app.core.logging import get_logger
from app.models.model_loader import model_store
from app.preprocessing.preprocess import (
    read_image_from_bytes, preprocess_for_classifier, preprocess_for_segmenter
)
from app.preprocessing.image_validation import validate_image_bytes
from app.inference.classifier_inference import run_classifier
from app.inference.segmentation_inference import run_segmenter
from app.inference.gradcam import generate_gradcam
from app.postprocessing.mask_processing import clean_mask, create_overlay
from app.utils.image_io import encode_image_to_base64, encode_mask_to_base64

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Service health and readiness check."""
    return HealthResponse(
        status="ok" if model_store.classifier_ready else "degraded",
        classifier_ready=model_store.classifier_ready,
        segmenter_ready=model_store.segmenter_ready,
        tensorflow_version=tf.__version__,
    )


@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Return model metadata."""
    return ModelInfoResponse(
        classifier={
            "name": "VGG16",
            "version": "1.0",
            "input_size": list(settings.img_size),
            "channels": settings.img_channels,
            "classes": ["no_tumor", "tumor"],
            "threshold": settings.classifier_threshold,
            "framework": "TensorFlow/Keras",
            "available": model_store.classifier_ready,
        },
        segmenter={
            "name": "U-Net",
            "version": "1.0",
            "input_size": [256, 256],
            "channels": 1,
            "framework": "TensorFlow/Keras",
            "available": model_store.segmenter_ready,
        },
    )


@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    Main MRI analysis endpoint.

    Runs the full pipeline:
    1. Validate uploaded file
    2. Brain crop + preprocess
    3. VGG-16 classification
    4. U-Net segmentation (if available)
    5. Grad-CAM
    6. Return structured JSON with base64 visualizations
    """
    # --- 1. Read and validate ---
    try:
        image_bytes = await file.read()
        validate_image_bytes(
            image_bytes=image_bytes,
            filename=file.filename or "upload",
            content_type=file.content_type or "image/jpeg",
            max_bytes=settings.max_file_size_bytes,
        )
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": {"code": "INVALID_FILE", "message": str(e)}},
        )

    # --- 2. Decode image ---
    try:
        img_bgr = read_image_from_bytes(image_bytes)
        original_shape = img_bgr.shape[:2]  # (H, W)
    except ValueError as e:
        return JSONResponse(
            status_code=422,
            content={"success": False, "error": {"code": "CORRUPTED_IMAGE", "message": str(e)}},
        )

    # --- 3. Classifier preprocessing ---
    try:
        clf_batch = preprocess_for_classifier(img_bgr)  # (1, 224, 224, 3)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": {"code": "CLASSIFICATION_FAILED", "message": "Preprocessing error"}},
        )

    # --- 4. Classification ---
    try:
        clf_result = run_classifier(clf_batch)
    except Exception as e:
        logger.error(f"Classification failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": {"code": "CLASSIFICATION_FAILED", "message": str(e)}},
        )

    # --- 5. Segmentation ---
    seg_result: dict[str, Any] = {"available": False, "mask": None, "tumor_pixels": None,
                  "tumor_percentage": None, "message": None}
    try:
        seg_batch = preprocess_for_segmenter(img_bgr)
        seg_result = run_segmenter(seg_batch, original_shape)

        # Post-process mask if available
        if seg_result["available"] and seg_result["mask"] is not None:
            seg_result["mask"] = clean_mask(seg_result["mask"])
    except Exception as e:
        logger.error(f"Segmentation failed: {e}", exc_info=True)
        seg_result["message"] = f"Segmentation error: {e}"

    # --- 6. Grad-CAM ---
    gradcam_heatmap = None
    gradcam_overlay = None
    try:
        gc = generate_gradcam(clf_batch, img_bgr)
        gradcam_heatmap = gc["heatmap"]
        gradcam_overlay = gc["overlay"]
    except Exception as e:
        logger.error(f"Grad-CAM failed: {e}", exc_info=True)

    # --- 7. Encode visualizations ---
    # Convert clf input back to displayable image (reverse VGG preprocess for display)
    display_img = img_bgr.copy()  # Use original BGR for display

    try:
        # Resize original to 224x224 for consistent display
        display_resized = cv2.resize(display_img, (224, 224), interpolation=cv2.INTER_CUBIC)

        viz_original = encode_image_to_base64(display_resized)

        viz_mask: str | None = None
        viz_overlay: str | None = None

        if seg_result["available"] and seg_result["mask"] is not None:
            mask = seg_result["mask"]
            mask_resized_display = cv2.resize(mask, (display_resized.shape[1], display_resized.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)
            viz_mask = encode_mask_to_base64(mask_resized_display)

            seg_overlay = create_overlay(display_resized, mask_resized_display)
            viz_overlay = encode_image_to_base64(seg_overlay)

        viz_gradcam: str | None = None
        viz_gradcam_overlay: str | None = None
        if gradcam_heatmap is not None:
            gc_resized = cv2.resize(gradcam_heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
            viz_gradcam = encode_image_to_base64(gc_resized)

        if gradcam_overlay is not None:
            gco_resized = cv2.resize(gradcam_overlay, (224, 224), interpolation=cv2.INTER_CUBIC)
            viz_gradcam_overlay = encode_image_to_base64(gco_resized)

    except Exception as e:
        logger.error(f"Visualization encoding failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": {"code": "INTERNAL_SERVER_ERROR", "message": "Visualization encoding failed"}},
        )

    # --- 8. Build response ---
    response = PredictResponse(
        success=True,
        analysis=AnalysisResult(
            prediction=clf_result["prediction"],
            confidence=clf_result["confidence"],
            raw_score=clf_result["raw_score"],
            segmentation=SegmentationResult(
                detected=seg_result["available"] and (seg_result["tumor_pixels"] or 0) > 0,
                tumor_pixels=seg_result["tumor_pixels"],
                tumor_percentage=seg_result["tumor_percentage"],
                physical_area_mm2=None,  # Not available for PNG/JPG without DICOM metadata
                message=seg_result.get("message"),
            ),
            visualizations=VisualizationsResult(
                original=viz_original,
                mask=viz_mask,
                overlay=viz_overlay,
                gradcam=viz_gradcam,
                gradcam_overlay=viz_gradcam_overlay,
            ),
            model=ModelInfo(
                classifier="VGG16",
                segmenter="U-Net",
                version="1.0",
                classifier_available=model_store.classifier_ready,
                segmenter_available=model_store.segmenter_ready,
            ),
        ),
    )

    logger.info(
        f"Prediction complete | {clf_result['prediction']} | "
        f"conf={clf_result['confidence']:.3f} | "
        f"seg={seg_result['available']}"
    )

    return response
