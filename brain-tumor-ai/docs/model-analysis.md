# Model Analysis & Evaluation

## VGG-16 Classifier
Based on the Kaggle `brain-tumor-images-for-brain-tumor-detection` dataset.

### Pipeline
- Input Size: (224, 224, 3)
- Base Architecture: Pre-trained VGG-16 (ImageNet weights, Top removed)
- Custom Head: Flatten -> Dropout(0.5) -> Dense(1, Sigmoid)
- Loss Function: Binary Crossentropy
- Optimizer: RMSprop (learning rate 1e-4)

### Data Preparation
The dataset images have widely varying dimensions. To standardise input:
1. Converted to grayscale and thresholded
2. Contour detection used to find extreme boundaries of the brain
3. Cropped to bounding box
4. Resized to 224x224
5. ImageNet mean subtraction applied (BGR format)

### Validation Metrics
- Validation Accuracy: ~90.8%
- Early Stopping triggered at ~16 epochs.

### Limitations & Caveats
- The test set defined in the notebook is extremely small (only 10 images).
- Patient-level splitting cannot be verified due to lack of metadata in the raw dataset. This introduces potential data leakage.

## U-Net Segmenter
A complementary model to locate the tumor visually.
- Input Size: (256, 256, 1) Grayscale
- Output: Binary mask mapped via a Sigmoid activation threshold (> 0.5)
- **Note**: A custom-trained checkpoint must be placed in `models/segmentation/unet_brain_tumor.h5` for this feature to work. If absent, the API degrades gracefully and reports the unavailability.
