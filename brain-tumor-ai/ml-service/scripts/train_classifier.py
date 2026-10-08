"""
Train the VGG-16 brain tumor classifier.

This script EXACTLY reproduces the Kaggle notebook training pipeline.
Run from the ml-service directory:

    python scripts/train_classifier.py \
        --data_dir /path/to/brain_tumor_dataset \
        --output_path models/classifier/vgg16_brain_tumor.h5 \
        --epochs 30

Dataset structure expected:
    brain_tumor_dataset/
    ├── no/   (no tumor)
    └── yes/  (tumor)

Output:
    models/classifier/vgg16_brain_tumor.h5

Training matches notebook:
  - VGG-16 pretrained ImageNet weights (top=False)
  - Dense(1, sigmoid) head
  - RMSprop lr=1e-4
  - binary_crossentropy loss
  - EarlyStopping(val_acc, patience=6)
  - crop_brain preprocessing
  - VGG-16 preprocess_input
  - Input size: (224, 224, 3)
"""
import argparse
import os
import shutil
import numpy as np
import cv2
import imutils
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

RANDOM_SEED = 123
IMG_SIZE = (224, 224)


def crop_brain(img: np.ndarray) -> np.ndarray:
    """Contour-based brain crop — matches notebook crop_imgs()."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if not cnts:
            return img
        c = max(cnts, key=cv2.contourArea)
        ext_left = tuple(c[c[:, :, 0].argmin()][0])
        ext_right = tuple(c[c[:, :, 0].argmax()][0])
        ext_top = tuple(c[c[:, :, 1].argmin()][0])
        ext_bot = tuple(c[c[:, :, 1].argmax()][0])
        y1, y2 = max(0, ext_top[1]), min(img.shape[0], ext_bot[1])
        x1, x2 = max(0, ext_left[0]), min(img.shape[1], ext_right[0])
        cropped = img[y1:y2, x1:x2].copy()
        return cropped if cropped.size > 0 else img
    except Exception:
        return img


def load_dataset(data_dir: str):
    """Load and preprocess dataset matching notebook pipeline."""
    X, y = [], []
    classes = sorted([d for d in os.listdir(data_dir) if not d.startswith(".")])
    label_map = {c: i for i, c in enumerate(classes)}
    print(f"Classes: {label_map}")

    for cls_name in classes:
        cls_dir = os.path.join(data_dir, cls_name)
        for fname in os.listdir(cls_dir):
            if fname.startswith("."):
                continue
            img_path = os.path.join(cls_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = crop_brain(img)
            img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
            X.append(img)
            y.append(label_map[cls_name])

    X = np.array(X, dtype=np.uint8)
    y = np.array(y, dtype=np.int32)
    print(f"Dataset loaded: {len(X)} images | Classes: {np.unique(y, return_counts=True)}")
    return X, y, label_map


def build_model():
    base = VGG16(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
    model = Sequential([
        base,
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.layers[0].trainable = False
    model.compile(loss="binary_crossentropy", optimizer=RMSprop(learning_rate=1e-4), metrics=["accuracy"])
    return model


def save_split_to_dirs(X, y, classes_inv, base_dir):
    """Save split images to class subdirectories for ImageDataGenerator."""
    for cls_name in classes_inv.values():
        os.makedirs(os.path.join(base_dir, cls_name.upper()), exist_ok=True)
    for i, (img, label) in enumerate(zip(X, y)):
        cls_name = classes_inv[label].upper()
        cv2.imwrite(os.path.join(base_dir, cls_name, f"{i}.jpg"), img)


def main():
    parser = argparse.ArgumentParser(description="Train VGG-16 brain tumor classifier")
    parser.add_argument("--data_dir", required=True, help="Path to brain_tumor_dataset/")
    parser.add_argument("--output_path", default="models/classifier/vgg16_brain_tumor.h5")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    print(f"TensorFlow: {tf.__version__}")
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    X, y, label_map = load_dataset(args.data_dir)
    classes_inv = {v: k for k, v in label_map.items()}

    # Split: matches notebook (80/10/10 approx)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.1, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1111, random_state=RANDOM_SEED, stratify=y_trainval
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Save to directories for ImageDataGenerator
    tmp_dir = "/tmp/brain_tumor_train"
    for split, X_s, y_s in [("train", X_train, y_train), ("val", X_val, y_val)]:
        save_split_to_dirs(X_s, y_s, classes_inv, os.path.join(tmp_dir, split))

    # Data generators — matches notebook augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        brightness_range=[0.5, 1.5],
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input,
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(tmp_dir, "train"), color_mode="rgb",
        target_size=IMG_SIZE, batch_size=args.batch_size,
        class_mode="binary", seed=RANDOM_SEED
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(tmp_dir, "val"), color_mode="rgb",
        target_size=IMG_SIZE, batch_size=16,
        class_mode="binary", seed=RANDOM_SEED
    )

    model = build_model()
    print(f"Model parameters: {model.count_params():,}")

    callbacks = [
        EarlyStopping(monitor="val_accuracy", mode="max", patience=6, restore_best_weights=True),
        ModelCheckpoint(args.output_path, monitor="val_accuracy", mode="max", save_best_only=True),
    ]

    history = model.fit(
        train_gen,
        steps_per_epoch=50,
        epochs=args.epochs,
        validation_data=val_gen,
        validation_steps=25,
        callbacks=callbacks,
    )

    # Evaluate on test set
    X_test_prep = np.array([preprocess_input(x.astype(np.float32)) for x in X_test])
    preds_raw = model.predict(X_test_prep)
    preds = [1 if p > 0.5 else 0 for p in preds_raw[:, 0]]
    test_acc = accuracy_score(y_test, preds)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(classification_report(y_test, preds, target_names=list(label_map.keys())))

    model.save(args.output_path)
    print(f"Model saved to {args.output_path}")

    # Cleanup temp dirs
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
