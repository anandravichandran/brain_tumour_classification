"""
Model loader — loads both classifier and segmenter at startup.

Models are loaded ONCE when FastAPI starts and held in memory.
Never reload per-request.

Handles missing checkpoint gracefully:
  - Classifier: Required. Service will not start without it (raises on startup).
  - Segmenter: Optional. Segmentation is skipped if checkpoint not found.
"""
from pathlib import Path
from typing import Optional
import tensorflow as tf

from app.models.classifier import build_classifier_architecture
from app.models.segmenter import build_unet
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ModelStore:
    """Singleton-style store for loaded models."""

    def __init__(self):
        self.classifier: Optional[tf.keras.Model] = None
        self.segmenter: Optional[tf.keras.Model] = None
        self.classifier_ready: bool = False
        self.segmenter_ready: bool = False

    def load_classifier(self) -> None:
        """Load VGG-16 classifier from .h5 checkpoint."""
        path = settings.classifier_path
        if not path.exists():
            raise FileNotFoundError(
                f"Classifier checkpoint not found at '{path}'. "
                f"Please train the model first using ml-service/scripts/train_classifier.py "
                f"and place the output at {path}"
            )

        logger.info(f"Loading classifier from {path}")
        # Load the full model (architecture + weights) saved via model.save()
        self.classifier = tf.keras.models.load_model(str(path), compile=False)
        self.classifier.trainable = False  # Ensure inference mode
        self.classifier_ready = True
        logger.info("Classifier loaded successfully")

    def load_segmenter(self) -> None:
        """Load U-Net segmenter from .h5 checkpoint. Non-fatal if missing."""
        path = settings.segmenter_path
        if not path.exists():
            logger.warning(
                f"Segmenter checkpoint not found at '{path}'. "
                f"Segmentation will be unavailable. "
                f"Train the segmenter using ml-service/scripts/train_segmenter.py"
            )
            self.segmenter_ready = False
            return

        logger.info(f"Loading segmenter from {path}")
        self.segmenter = tf.keras.models.load_model(str(path), compile=False)
        self.segmenter.trainable = False
        self.segmenter_ready = True
        logger.info("Segmenter loaded successfully")

    def load_all(self) -> None:
        """Load all models. Called once on FastAPI startup."""
        logger.info(f"TensorFlow version: {tf.__version__}")
        logger.info(f"GPU devices: {tf.config.list_physical_devices('GPU')}")

        self.load_classifier()  # Raises if missing (required)
        self.load_segmenter()   # Warning only if missing (optional)

        logger.info(
            f"Model store ready | classifier={self.classifier_ready} | "
            f"segmenter={self.segmenter_ready}"
        )


# Global singleton — imported everywhere
model_store = ModelStore()
