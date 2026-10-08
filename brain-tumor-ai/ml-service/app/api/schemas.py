"""
Pydantic schemas for FastAPI request/response validation.
These are API contracts — NOT model storage.
"""
from pydantic import BaseModel, Field
from typing import Optional


class SegmentationResult(BaseModel):
    detected: bool
    tumor_pixels: Optional[int] = None
    tumor_percentage: Optional[float] = None
    physical_area_mm2: Optional[float] = None  # Only for DICOM with pixel spacing
    message: Optional[str] = None  # Explains if unavailable


class VisualizationsResult(BaseModel):
    original: str = Field(description="Base64-encoded original preprocessed image")
    mask: Optional[str] = Field(None, description="Base64-encoded binary segmentation mask")
    overlay: Optional[str] = Field(None, description="Base64-encoded segmentation overlay")
    gradcam: Optional[str] = Field(None, description="Base64-encoded Grad-CAM heatmap")
    gradcam_overlay: Optional[str] = Field(None, description="Base64-encoded Grad-CAM overlay on original")


class ModelInfo(BaseModel):
    classifier: str = "VGG16"
    segmenter: str = "U-Net"
    version: str = "1.0"
    classifier_available: bool
    segmenter_available: bool


class AnalysisResult(BaseModel):
    prediction: str = Field(description="'tumor' or 'no_tumor'")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the prediction")
    raw_score: float = Field(ge=0.0, le=1.0, description="Raw sigmoid output")
    segmentation: SegmentationResult
    visualizations: VisualizationsResult
    model: ModelInfo


class PredictResponse(BaseModel):
    success: bool = True
    analysis: AnalysisResult


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str
    classifier_ready: bool
    segmenter_ready: bool
    tensorflow_version: str


class ModelInfoResponse(BaseModel):
    classifier: dict
    segmenter: dict
