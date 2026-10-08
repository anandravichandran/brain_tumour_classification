"""
Application configuration using Pydantic Settings.
Reads from environment variables / .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=("settings_",)
    )

    # Server
    port: int = 8000
    allowed_origins: str = "http://localhost:3000,http://localhost:5173"

    # Model paths (relative to ml-service root)
    model_path: str = "models/classifier/vgg16_brain_tumor.h5"
    segmentation_model_path: str = "models/segmentation/unet_brain_tumor.h5"

    # Inference settings
    device: str = "cpu"  # 'cpu' or 'cuda'
    classifier_threshold: float = 0.5
    max_file_size_mb: int = 10

    # Image settings — MUST match notebook training
    img_size: tuple = (224, 224)
    img_channels: int = 3

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def classifier_path(self) -> Path:
        return Path(self.model_path)

    @property
    def segmenter_path(self) -> Path:
        return Path(self.segmentation_model_path)


settings = Settings()
