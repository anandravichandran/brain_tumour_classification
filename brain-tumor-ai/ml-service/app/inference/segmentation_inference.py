"""
U-Net Segmentation Inference.

If the segmenter is not available, returns a structured response
indicating segmentation is unavailable — never returns fake results.
"""
import numpy as np
import cv2
from app.models.model_loader import model_store
from app.core.logging import get_logger

logger = get_logger(__name__)

SEGMENTATION_THRESHOLD = 0.5  # Binary threshold for probability mask


def run_segmenter(
    preprocessed_seg_batch: np.ndarray,
    original_shape: tuple,
) -> dict:
    """
    Run U-Net segmentation inference.

    Args:
        preprocessed_seg_batch: float32 (1, 256, 256, 1) — grayscale [0,1]
        original_shape: (H, W) of original image for mask upscaling

    Returns:
        dict with:
          - available: bool
          - mask: np.ndarray binary (H, W) uint8 or None
          - tumor_pixels: int or None
          - tumor_percentage: float or None
          - probability_map: np.ndarray (256, 256) float32 or None
    """
    if not model_store.segmenter_ready:
        logger.warning("Segmenter not available — skipping segmentation")
        return {
            "available": False,
            "mask": None,
            "tumor_pixels": None,
            "tumor_percentage": None,
            "probability_map": None,
            "message": "Segmentation model checkpoint requires validation/training.",
        }

    # The current segmenter checkpoint was generated via scripts/init_models.py
    # which only initialized random weights but never trained the model.
    # Do NOT fake segmentation.
    logger.warning("Segmenter checkpoint is untrained — skipping segmentation")
    return {
        "available": False,
        "mask": None,
        "tumor_pixels": None,
        "tumor_percentage": None,
        "probability_map": None,
        "message": "Segmentation model checkpoint requires validation/training.",
    }
