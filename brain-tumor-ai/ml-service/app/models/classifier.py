"""
VGG-16 Classifier definition — matches Kaggle notebook architecture exactly.

Architecture:
    Sequential([
        VGG16(weights=..., include_top=False, input_shape=(224,224,3)),
        Flatten(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

Notes:
  - Base VGG-16 is loaded WITHOUT top (include_top=False)
  - Single sigmoid output (binary classification)
  - Inference threshold: 0.5 (from notebook)
  - Class 0: NO tumor | Class 1: YES tumor
"""
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from app.core.logging import get_logger

logger = get_logger(__name__)

IMG_SIZE = (224, 224)
NUM_CLASSES = 1  # Binary → single sigmoid output


def build_classifier_architecture() -> Sequential:
    """
    Rebuild the exact model architecture from the Kaggle notebook.
    Used when loading weights from a checkpoint or for training.
    """
    base_model = VGG16(
        weights=None,           # weights loaded separately from .h5
        include_top=False,
        input_shape=IMG_SIZE + (3,)
    )

    model = Sequential([
        base_model,
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="sigmoid"),
    ])

    # Freeze base (matches training setup — base was frozen during training)
    model.layers[0].trainable = False

    model.compile(
        loss="binary_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy"],
    )

    logger.info("Classifier architecture built (VGG-16 + head)")
    return model


def get_last_conv_layer_name(model: Sequential) -> str:
    """
    Identify the last convolutional layer in VGG-16 for Grad-CAM.
    VGG-16 last conv layer: 'block5_conv3'
    """
    # The base VGG-16 model is at model.layers[0]
    base = model.layers[0]
    for layer in reversed(base.layers):
        if isinstance(layer, layers.Conv2D):
            logger.info(f"Last conv layer identified: {layer.name}")
            return layer.name
    raise RuntimeError("Could not find last conv layer in VGG-16")
