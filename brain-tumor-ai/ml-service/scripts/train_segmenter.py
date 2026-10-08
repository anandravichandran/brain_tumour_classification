"""
Train the U-Net brain tumor segmentation model.

Uses the BRATS or brain MRI segmentation dataset.
Run from ml-service directory:

    python scripts/train_segmenter.py \
        --data_dir /path/to/segmentation_dataset \
        --output_path models/segmentation/unet_brain_tumor.h5 \
        --epochs 50

Dataset structure expected (BRATS-style):
    segmentation_dataset/
    ├── images/   (MRI .jpg/.png files)
    └── masks/    (Binary mask .jpg/.png — same filenames as images)

Alternative: Use Kaggle "Brain MRI Images for Brain Tumor Detection" with manual masks.

Input: Grayscale MRI (256×256×1) normalized to [0, 1]
Output: Binary mask (256×256×1) sigmoid probability
"""
import argparse
import os
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.models.segmenter import build_unet

RANDOM_SEED = 123
IMG_SIZE = (256, 256)


def load_segmentation_dataset(data_dir: str):
    """Load paired MRI images and segmentation masks."""
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")

    X, y = [], []
    for fname in sorted(os.listdir(images_dir)):
        if fname.startswith("."):
            continue
        img_path = os.path.join(images_dir, fname)
        mask_path = os.path.join(masks_dir, fname)

        if not os.path.exists(mask_path):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            continue

        img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)  # Binary mask

        X.append(img[..., np.newaxis])    # (256, 256, 1)
        y.append(mask[..., np.newaxis])   # (256, 256, 1)

    X = np.array(X)
    y = np.array(y)
    print(f"Segmentation dataset: {len(X)} pairs")
    return X, y


def dice_coefficient(y_true, y_pred, smooth=1.0):
    """Dice coefficient metric."""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + tf.keras.losses.binary_crossentropy(y_true, y_pred)


def main():
    parser = argparse.ArgumentParser(description="Train U-Net brain tumor segmenter")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_path", default="models/segmentation/unet_brain_tumor.h5")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    print(f"TensorFlow: {tf.__version__}")
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    X, y = load_segmentation_dataset(args.data_dir)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    model = build_unet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=combined_loss,
        metrics=[dice_coefficient, "accuracy"],
    )

    callbacks = [
        EarlyStopping(monitor="val_dice_coefficient", mode="max", patience=10, restore_best_weights=True),
        ModelCheckpoint(args.output_path, monitor="val_dice_coefficient", mode="max", save_best_only=True),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    model.save(args.output_path)
    print(f"Segmenter saved to {args.output_path}")


if __name__ == "__main__":
    main()
