"""
Grad-CAM implementation for VGG-16 classifier.

Target layer: 'block5_conv3' — the last convolutional layer in VGG-16.
This provides the best spatial resolution for localization.

Algorithm:
1. Create a sub-model: inputs → last_conv_outputs + final_predictions
2. Forward pass with GradientTape
3. Compute gradients of prediction score w.r.t. conv feature maps
4. Global average pool the gradients → per-channel weights
5. Weighted sum of feature maps → activation map
6. ReLU → resize to input dimensions
7. Normalize to [0, 255] → apply colormap → overlay on original
"""
import numpy as np
import cv2
import tensorflow as tf
from app.models.model_loader import model_store
from app.core.logging import get_logger

logger = get_logger(__name__)

GRADCAM_LAYER = "block5_conv3"  # VGG-16 last conv layer


def _build_gradcam_models() -> tuple[tf.keras.Model, tf.keras.Model]:
    """
    Build two models:
      1. last_conv_model: inputs -> last conv layer activations
      2. head_model: last conv activations -> final prediction
    Compatible with Keras 3 and Sequential/Functional wrapper architectures.
    """
    classifier = model_store.classifier
    vgg_base = classifier.layers[0]

    last_conv_layer = None
    try:
        last_conv_layer = vgg_base.get_layer(GRADCAM_LAYER)
    except ValueError:
        for layer in reversed(vgg_base.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                logger.warning(f"Using fallback conv layer: {layer.name}")
                break

    if last_conv_layer is None:
        raise ValueError(f"Could not find target layer {GRADCAM_LAYER} or any Conv2D layer in base model")

    # 1. Inputs to conv layer
    last_conv_model = tf.keras.Model(vgg_base.inputs, last_conv_layer.output)

    # 2. Conv layer to classifier output
    conv_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = conv_input

    # Remaining layers of vgg_base after last_conv_layer
    found = False
    for layer in vgg_base.layers:
        if layer.name == last_conv_layer.name:
            found = True
            continue
        if found:
            x = layer(x)

    # Subsequent top layers in Sequential classifier (Flatten, Dropout, Dense)
    for layer in classifier.layers[1:]:
        x = layer(x)

    head_model = tf.keras.Model(conv_input, x)

    return last_conv_model, head_model


def generate_gradcam(
    preprocessed_batch: np.ndarray,
    original_img_bgr: np.ndarray,
) -> dict:
    """
    Generate Grad-CAM heatmap and overlay.

    Args:
        preprocessed_batch: (1, 224, 224, 3) float32 — classifier input
        original_img_bgr: (H, W, 3) uint8 BGR — for overlay

    Returns:
        dict with:
          - heatmap: np.ndarray (H, W, 3) uint8 BGR — colorized heatmap
          - overlay: np.ndarray (H, W, 3) uint8 BGR — alpha-blended overlay
          - heatmap_raw: np.ndarray (7, 7) float32 — raw activation map
    """
    if not model_store.classifier_ready:
        raise RuntimeError("Classifier not loaded — cannot generate Grad-CAM")

    try:
        last_conv_model, head_model = _build_gradcam_models()
        input_tensor = tf.cast(preprocessed_batch, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs = last_conv_model(input_tensor)
            tape.watch(conv_outputs)
            predictions = head_model(conv_outputs)
            pred_score = predictions[:, 0]

        # Gradients of pred_score w.r.t. conv outputs
        grads = tape.gradient(pred_score, conv_outputs)  # (1, H_conv, W_conv, C)

        # Pool gradients over spatial dimensions → per-channel weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

        # Weight the conv output channels
        conv_out = conv_outputs[0]  # (H_conv, W_conv, C)
        heatmap = conv_out @ pooled_grads[..., tf.newaxis]  # (H_conv, W_conv, 1)
        heatmap = tf.squeeze(heatmap)  # (H_conv, W_conv)

        # ReLU
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap.numpy()

        # Normalize to [0, 1]
        heatmap_max = np.max(heatmap)
        if heatmap_max > 0:
            heatmap = heatmap / heatmap_max

        heatmap_raw = heatmap.copy()

        # Resize to original image dimensions
        orig_h, orig_w = original_img_bgr.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))

        # Apply colormap (COLORMAP_JET: blue→green→red)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Overlay: alpha-blend colormap on original image
        overlay = cv2.addWeighted(original_img_bgr, 0.6, heatmap_colored, 0.4, 0)

        logger.info(f"Grad-CAM generated | layer={GRADCAM_LAYER} | heatmap_shape={heatmap_raw.shape}")

        return {
            "heatmap": heatmap_colored,
            "overlay": overlay,
            "heatmap_raw": heatmap_raw,
        }

    except Exception as e:
        logger.error(f"Grad-CAM failed: {e}", exc_info=True)
        raise RuntimeError(f"Grad-CAM generation failed: {e}")
