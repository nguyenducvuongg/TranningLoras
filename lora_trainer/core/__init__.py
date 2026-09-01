"""
Core module for TranningLoras.
Chứa các thành phần cốt lõi: Registry, Hardware detection, Base Engine, Key Vault.
"""

from .model_registry import (
    MODEL_REGISTRY,
    VAE_REGISTRY,
    VAE_FALLBACKS,
    TEXT_ENCODER_REGISTRY,
    TEXT_ENCODER_FALLBACKS,
    get_model_info,
    get_preferred_engine,
    list_supported_models,
)
from .hardware import detect_hardware_environment, setup_cuda_environment
from .key_vault import (
    save_api_key,
    get_api_key,
    load_api_vault,
    display_key_vault_dashboard,
    get_vault_path,
)
from .base_engine import BaseTrainerEngine

__all__ = [
    "MODEL_REGISTRY",
    "VAE_REGISTRY",
    "VAE_FALLBACKS",
    "TEXT_ENCODER_REGISTRY",
    "TEXT_ENCODER_FALLBACKS",
    "get_model_info",
    "get_preferred_engine",
    "list_supported_models",
    "detect_hardware_environment",
    "setup_cuda_environment",
    "save_api_key",
    "get_api_key",
    "load_api_vault",
    "display_key_vault_dashboard",
    "get_vault_path",
    "BaseTrainerEngine",
]
