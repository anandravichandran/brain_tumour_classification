"""
Initialize default model weights for Classifier (VGG-16) and Segmenter (U-Net).
Enables the application to boot and serve inference immediately.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from app.models.segmenter import build_unet

def main():
    base_dir = Path(__file__).resolve().parent.parent
    classifier_dir = base_dir / "models" / "classifier"
    segmenter_dir = base_dir / "models" / "segmentation"
    classifier_dir.mkdir(parents=True, exist_ok=True)
    segmenter_dir.mkdir(parents=True, exist_ok=True)

    classifier_path = classifier_dir / "vgg16_brain_tumor.h5"
    segmenter_path = segmenter_dir / "unet_brain_tumor.h5"

    print("--- 1. Building VGG-16 Classifier ---")
    base_vgg = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_vgg.trainable = False

    classifier = Sequential([
        base_vgg,
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])

    classifier.compile(
        loss="binary_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy"]
    )
    print(f"Saving classifier to {classifier_path} ...")
    classifier.save(str(classifier_path))
    print(f"Classifier saved successfully ({classifier_path.stat().st_size / (1024*1024):.2f} MB)")

    print("--- 2. Building U-Net Segmenter ---")
    segmenter = build_unet(input_size=(256, 256), channels=1)
    print(f"Saving segmenter to {segmenter_path} ...")
    segmenter.save(str(segmenter_path))
    print(f"Segmenter saved successfully ({segmenter_path.stat().st_size / (1024*1024):.2f} MB)")

    print("\nAll models initialized successfully!")

if __name__ == "__main__":
    main()
