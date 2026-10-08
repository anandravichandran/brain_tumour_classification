"""
Preprocessing pipeline that EXACTLY matches the Kaggle notebook training pipeline.

Pipeline:
  1. Read image (handle BGR from OpenCV)
  2. crop_brain() — contour-based extreme-point crop (matches notebook crop_imgs)
  3. Resize to (224, 224) with INTER_CUBIC
  4. Apply keras.applications.vgg16.preprocess_input (subtracts ImageNet BGR means)
  5. Expand dims to (1, 224, 224, 3)

This preprocessing is authoritative — do NOT alter it without retraining.
"""
import cv2
import numpy as np
import imutils
from app.core.logging import get_logger

logger = get_logger(__name__)


def crop_brain(img: np.ndarray, add_pixels: int = 0) -> np.ndarray:
    """
    Crop the brain out of an MRI scan using contour-based extreme-point detection.
    Matches notebook's crop_imgs() function exactly.

    Args:
        img: Input image in BGR or RGB format (HxWxC uint8)
        add_pixels: Extra pixels to add around bounding box

    Returns:
        Cropped image, or original image if crop fails
    """
    try:
        # Convert to grayscale (from RGB/BGR — works either way)
        if img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours — grab largest
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        if not cnts:
            logger.warning("crop_brain: no contours found, returning original image")
            return img

        c = max(cnts, key=cv2.contourArea)

        # Extreme points
        ext_left = tuple(c[c[:, :, 0].argmin()][0])
        ext_right = tuple(c[c[:, :, 0].argmax()][0])
        ext_top = tuple(c[c[:, :, 1].argmin()][0])
        ext_bot = tuple(c[c[:, :, 1].argmax()][0])

        # Crop with safety bounds
        y1 = max(0, ext_top[1] - add_pixels)
        y2 = min(img.shape[0], ext_bot[1] + add_pixels)
        x1 = max(0, ext_left[0] - add_pixels)
        x2 = min(img.shape[1], ext_right[0] + add_pixels)

        cropped = img[y1:y2, x1:x2].copy()

        if cropped.size == 0:
            logger.warning("crop_brain: empty crop result, returning original")
            return img

        return cropped

    except Exception as e:
        logger.warning(f"crop_brain failed ({e}), returning original image")
        return img


def vgg16_preprocess_input(img: np.ndarray) -> np.ndarray:
    """
    Apply VGG-16 preprocessing identical to keras.applications.vgg16.preprocess_input.
    Subtracts ImageNet BGR channel means:
        Blue: 103.939, Green: 116.779, Red: 123.68

    IMPORTANT: VGG16 preprocess_input expects BGR input (as produced by cv2.imread).
    The notebook uses cv2.imread (BGR) and passes directly to preprocess_input.
    """
    img = img.astype(np.float32)
    # Subtract BGR means (same as Keras VGG16 preprocess_input in 'caffe' mode)
    img[..., 0] -= 103.939  # Blue channel
    img[..., 1] -= 116.779  # Green channel
    img[..., 2] -= 123.68   # Red channel
    return img


def preprocess_for_classifier(img_bgr: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline for the VGG-16 classifier.
    Input: BGR image as loaded by cv2.imread (uint8, HxWxC)
    Output: float32 numpy array of shape (1, 224, 224, 3)
    """
    # Step 1: Brain crop (contour-based)
    cropped = crop_brain(img_bgr)

    # Step 2: Resize to 224×224
    resized = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_CUBIC)

    # Step 3: VGG-16 preprocess_input (subtract BGR means)
    preprocessed = vgg16_preprocess_input(resized)

    # Step 4: Add batch dimension
    batch = np.expand_dims(preprocessed, axis=0)  # (1, 224, 224, 3)

    return batch


def preprocess_for_segmenter(img_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocessing for U-Net segmentation model.
    Input: BGR image (uint8, HxWxC)
    Output: float32 tensor (1, 256, 256, 1) — grayscale normalized to [0, 1]
    """
    # Convert to grayscale for segmentation
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Resize to segmentation input size (256×256)
    resized = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_CUBIC)

    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    # Add channel and batch dims: (1, 256, 256, 1)
    batch = np.expand_dims(np.expand_dims(normalized, axis=-1), axis=0)

    return batch


def read_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Decode uploaded image bytes to BGR numpy array.
    Returns uint8 BGR image.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # Always returns BGR
    if img is None:
        raise ValueError("Failed to decode image — file may be corrupt or unsupported format")
    return img
