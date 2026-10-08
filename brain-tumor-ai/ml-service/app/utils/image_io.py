"""
Image encoding utilities for API responses.
Converts numpy arrays to base64-encoded strings for JSON transport.
"""
import base64
import cv2
import numpy as np
from app.core.logging import get_logger

logger = get_logger(__name__)


def encode_image_to_base64(img: np.ndarray, fmt: str = ".jpg", quality: int = 90) -> str:
    """
    Encode a numpy image array to base64 string.

    Args:
        img: numpy array (H, W, C) uint8 BGR or grayscale
        fmt: output format '.jpg' or '.png'
        quality: JPEG quality (1-100), ignored for PNG

    Returns:
        Base64-encoded string (without data URI prefix)
    """
    params = []
    if fmt.lower() in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]

    success, buffer = cv2.imencode(fmt, img, params)
    if not success:
        raise ValueError(f"Failed to encode image as {fmt}")

    return base64.b64encode(buffer).decode("utf-8")


def encode_mask_to_base64(mask: np.ndarray) -> str:
    """
    Encode a binary mask (uint8, 0/255) to base64 PNG.
    """
    return encode_image_to_base64(mask, fmt=".png")
