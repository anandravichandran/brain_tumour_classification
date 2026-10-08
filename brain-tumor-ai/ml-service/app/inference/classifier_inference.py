"""
VGG-16 Classification Inference.

Runs the trained classifier on a preprocessed image tensor.
Returns prediction class and confidence score.

Class mapping (alphabetical directory sort from notebook):
  0 → NO (no tumor)
  1 → YES (tumor present)

Threshold: 0.5 (from notebook: `1 if x > 0.5 else 0`)
"""
import numpy as np
from app.models.model_loader import model_store
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

CLASS_NAMES = {0: "no_tumor", 1: "tumor"}


def run_classifier(preprocessed_batch: np.ndarray) -> dict:
    """
    Run VGG-16 classifier inference.

    Args:
        preprocessed_batch: float32 array (1, 224, 224, 3)
            preprocessed exactly as in preprocess_for_classifier()

    Returns:
        dict with:
          - prediction: "tumor" | "no_tumor"
          - confidence: float [0.0, 1.0]
          - raw_score: float (sigmoid output)
    """
    if not model_store.classifier_ready:
        raise RuntimeError("Classifier model is not loaded")

    raw_scores = model_store.classifier.predict(preprocessed_batch, verbose=0)
    # raw_scores shape: (1, 1) — single sigmoid output
    raw_score = float(raw_scores[0][0])

    # Apply threshold (matches notebook: 1 if x > 0.5 else 0)
    predicted_class = 1 if raw_score > settings.classifier_threshold else 0
    prediction_label = CLASS_NAMES[predicted_class]

    # Confidence: probability of the predicted class
    confidence = raw_score if predicted_class == 1 else (1.0 - raw_score)

    logger.info(
        f"Classification: {prediction_label} | raw={raw_score:.4f} | confidence={confidence:.4f}"
    )

    return {
        "prediction": prediction_label,
        "confidence": round(confidence, 4),
        "raw_score": round(raw_score, 4),
        "predicted_class_idx": predicted_class,
    }
