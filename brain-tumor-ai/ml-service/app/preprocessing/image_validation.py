"""
Image validation utilities.
"""
import numpy as np
import cv2
from app.core.logging import get_logger

logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
MAX_DIMENSION = 4096
MIN_DIMENSION = 32


def validate_image_bytes(image_bytes: bytes, filename: str, content_type: str, max_bytes: int) -> None:
    """
    Validate uploaded image file.

    Raises:
        ValueError with descriptive message on any validation failure.
    """
    import os
    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    if content_type not in ALLOWED_MIME_TYPES:
        raise ValueError(f"Unsupported MIME type '{content_type}'. Allowed: image/jpeg, image/png")

    if len(image_bytes) > max_bytes:
        max_mb = max_bytes / (1024 * 1024)
        raise ValueError(f"File too large. Maximum allowed: {max_mb:.1f} MB")

    if len(image_bytes) == 0:
        raise ValueError("Uploaded file is empty")

    # Attempt decode
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("File appears to be corrupt or is not a valid image")

    h, w = img.shape[:2]
    if h < MIN_DIMENSION or w < MIN_DIMENSION:
        raise ValueError(f"Image too small ({w}×{h}). Minimum: {MIN_DIMENSION}×{MIN_DIMENSION} px")

    if h > MAX_DIMENSION or w > MAX_DIMENSION:
        raise ValueError(f"Image too large ({w}×{h}). Maximum: {MAX_DIMENSION}×{MAX_DIMENSION} px")
