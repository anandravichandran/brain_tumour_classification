"""
FastAPI application entry point.

Startup sequence:
1. Load VGG-16 classifier (required)
2. Load U-Net segmenter (optional)
3. Register routes
4. Start serving
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.core.logging import get_logger
from app.models.model_loader import model_store

logger = get_logger("brain_tumor_ml")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    logger.info("Starting Brain Tumor ML Service...")
    logger.info(f"Classifier path: {settings.classifier_path}")
    logger.info(f"Segmenter path: {settings.segmenter_path}")

    try:
        model_store.load_all()
        logger.info("All models loaded. Service ready.")
    except FileNotFoundError as e:
        # Critical: classifier is required
        logger.critical(f"STARTUP FAILED — Model not found: {e}")
        raise RuntimeError(str(e))

    yield

    logger.info("Shutting down Brain Tumor ML Service...")


app = FastAPI(
    title="Brain Tumor Detection & Segmentation — ML Service",
    description=(
        "Academic AI system for brain tumor detection and segmentation using MRI images. "
        "NOT intended for clinical use."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router, prefix="")

logger.info("FastAPI application configured")
