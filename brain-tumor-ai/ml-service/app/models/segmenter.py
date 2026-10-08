"""
U-Net Segmentation Model Architecture.

This is a standard U-Net adapted for brain tumor segmentation on grayscale MRI.
Input: (batch, 256, 256, 1) — grayscale normalized to [0, 1]
Output: (batch, 256, 256, 1) — sigmoid probability mask

IMPORTANT: This architecture must match the checkpoint that was used for training.
If training a fresh model, use the train_segmenter.py script.
"""
from tensorflow.keras import layers, Model
from tensorflow.keras import Input
from app.core.logging import get_logger

logger = get_logger(__name__)

UNET_IMG_SIZE = (256, 256)
UNET_CHANNELS = 1


def conv_block(x, filters: int, kernel_size: int = 3):
    """Double convolution block used in U-Net."""
    x = layers.Conv2D(filters, kernel_size, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, kernel_size, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x


def encoder_block(x, filters: int):
    """Encoder block: conv_block + max pool. Returns (skip, pooled)."""
    skip = conv_block(x, filters)
    pooled = layers.MaxPooling2D((2, 2))(skip)
    return skip, pooled


def decoder_block(x, skip, filters: int):
    """Decoder block: upsample + concatenate skip + conv_block."""
    x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x


def build_unet(input_size: tuple = UNET_IMG_SIZE, channels: int = UNET_CHANNELS) -> Model:
    """
    Build U-Net model for binary brain tumor segmentation.

    Args:
        input_size: (H, W) of input images
        channels: number of input channels (1 for grayscale)

    Returns:
        Compiled Keras Model
    """
    inputs = Input(shape=input_size + (channels,))

    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bottleneck
    b = conv_block(p4, 1024)

    # Decoder
    d1 = decoder_block(b, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # Output
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="UNet_BrainTumor")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    logger.info(f"U-Net architecture built: input={input_size+(channels,)}")
    return model
