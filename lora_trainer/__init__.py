"""
TranningLoras: Universal & Optimized LoRA Training Suite
Tích hợp các Engine huấn luyện hàng đầu thế giới:
- Kohya sd-scripts (SDXL, Pony, Illustrious, SD1.5, SD3.5)
- Kohya Musubi-Tuner (Wan 2.1/2.2, FLUX.2 Klein, Qwen-Image, Z-Image, Krea2)
- Ostris AI-Toolkit (FLUX.1 dev/schnell/kontext, SDXL)
"""

__version__ = "2.0.0"

from .core.model_registry import (
    MODEL_REGISTRY,
    VAE_REGISTRY,
    TEXT_ENCODER_REGISTRY,
    get_model_info,
    get_preferred_engine,
    list_supported_models,
)
from .core.hardware import detect_hardware_environment, setup_cuda_environment
from .core.key_vault import save_api_key, get_api_key, display_key_vault_dashboard
from .storage.drive_manager import setup_storage_structure
from .storage.downloader import aria2_download
from .storage.model_fetcher import download_model_suite
from .engines.unified_trainer import run_unified_training
from .dataset.builder import build_dataset_list
from .dataset.cleaner import clean_directory
from .dataset.renamer import batch_standardize_datasets
from .captioning.gemini import batch_caption_gemini
from .ui.dashboard import TrainingDashboard, get_dashboard

__all__ = [
    "__version__",
    "MODEL_REGISTRY",
    "VAE_REGISTRY",
    "TEXT_ENCODER_REGISTRY",
    "get_model_info",
    "get_preferred_engine",
    "list_supported_models",
    "detect_hardware_environment",
    "setup_cuda_environment",
    "save_api_key",
    "get_api_key",
    "display_key_vault_dashboard",
    "setup_storage_structure",
    "aria2_download",
    "download_model_suite",
    "run_unified_training",
    "build_dataset_list",
    "clean_directory",
    "batch_standardize_datasets",
    "batch_caption_gemini",
    "TrainingDashboard",
    "get_dashboard",
]
