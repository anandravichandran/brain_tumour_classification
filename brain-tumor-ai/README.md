# Brain Tumor Detection & Segmentation AI

An academic AI system for analyzing MRI images to detect and segment brain tumors.

## Architecture

The system follows a stateless, three-tier architecture:
1. **Frontend**: React, TypeScript, Vite, TailwindCSS
2. **Backend Gateway**: Node.js, Express, TypeScript (handles API routing, validation, DeepSeek AI)
3. **ML Inference Service**: Python, FastAPI, TensorFlow/Keras, OpenCV

## Model Pipelines

### 1. Classification (VGG-16)
- Identifies if a tumor is present or not.
- **Preprocessing**: Contour-based extreme-point cropping, resized to 224x224, ImageNet BGR mean subtraction.
- **Architecture**: Frozen VGG-16 base + Dense classifier head.

### 2. Segmentation (U-Net)
- Maps the exact tumor pixels.
- **Preprocessing**: Grayscale conversion, resized to 256x256, normalized to [0, 1].
- **Architecture**: Encoder-decoder U-Net with skip connections.

### 3. Grad-CAM
- Provides visual attention heatmaps indicating which regions influenced the VGG-16 model's classification.

## Deployment

This repository is configured for:
- **Frontend**: Vercel
- **Backend (Node)**: Render
- **ML Service (Python)**: Render (using tensorflow-cpu)

For detailed deployment instructions, see `docs/deployment.md`.

## Local Development

You can run the full stack locally via Docker Compose:

```bash
docker-compose up --build
```
- Frontend: http://localhost:5173
- Backend: http://localhost:3001
- ML Service: http://localhost:8000

## Academic Disclaimer
This system is intended for **educational and research purposes only**. It is not a medical diagnostic device and must not be used as a substitute for evaluation by a qualified medical professional.
