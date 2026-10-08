# System Architecture

## Overview
The Brain Tumor Detection & Segmentation AI is built with a decoupled, three-tier architecture ensuring scalability, statelessness, and distinct separation of concerns.

## 1. Frontend (React/Vite)
- **Framework:** React 18, Vite, TypeScript
- **Styling:** TailwindCSS with custom glassmorphism utilities for a premium, academic aesthetic.
- **Responsibility:** Handles user file uploads, visualizes returned base64-encoded heatmaps/masks, and formats AI explanations.

## 2. API Gateway (Node.js/Express)
- **Framework:** Express, TypeScript
- **Responsibility:** Acts as a proxy between the public internet and the ML service.
- **Features:**
  - Memory-based Multer limits file sizes without hitting disk.
  - Zod runtime validation ensures strict API contracts between Node and Python.
  - DeepSeek API integration adds a human-readable explanation *after* ML inference is complete.

## 3. ML Inference Service (Python/FastAPI)
- **Framework:** FastAPI, TensorFlow 2.x, OpenCV
- **Responsibility:** The core brain of the application.
- **Features:**
  - Precise preprocessing (contour cropping, ImageNet norm) mirroring Kaggle notebooks exactly.
  - VGG-16 inference for binary classification.
  - U-Net inference for binary segmentation (degrades gracefully if model missing).
  - Grad-CAM heatmap generation to visualize network attention.
  - Stateless execution: images are processed entirely in memory or temporary buffers and returned as base64.
