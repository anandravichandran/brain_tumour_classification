"""
Mask post-processing utilities.
"""
import numpy as np
import cv2
from app.core.logging import get_logger

logger = get_logger(__name__)


def clean_mask(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    """
    Remove small noisy regions from binary mask using connected component analysis.

    Args:
        mask: Binary mask (H, W) uint8 — values 0 or 255
        min_area: Minimum component area in pixels to keep

    Returns:
        Cleaned binary mask (H, W) uint8
    """
    cleaned = np.zeros_like(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned


def create_overlay(
    original_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple = (0, 0, 255),  # Red in BGR
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Blend the tumor mask onto the original MRI image.

    Args:
        original_bgr: Original image (H, W, 3) uint8 BGR
        mask: Binary mask (H, W) uint8 — values 0 or 255
        color_bgr: Color for tumor region in BGR
        alpha: Transparency of overlay (0=transparent, 1=opaque)

    Returns:
        Overlay image (H, W, 3) uint8 BGR
    """
    overlay = original_bgr.copy()
    tumor_region = mask > 0
    overlay[tumor_region] = (
        (1 - alpha) * original_bgr[tumor_region] + alpha * np.array(color_bgr)
    ).astype(np.uint8)
    return overlay
