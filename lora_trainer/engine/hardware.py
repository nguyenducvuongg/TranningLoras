"""
Hardware & GPU Detection
Tự động phát hiện loại card GPU, kiến trúc CUDA và tối ưu các biến môi trường cho PyTorch trên Google Colab.
"""

import os
from typing import Dict, Any


def setup_cuda_environment() -> str:
    """Tự động thiết lập TORCH_CUDA_ARCH_LIST phù hợp với GPU Colab."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"

        gpu_name = torch.cuda.get_device_name(0)
        arch = "7.5" # default T4

        if any(k in gpu_name for k in ["A100", "A10", "A30"]):
            arch = "8.0"
        elif any(k in gpu_name for k in ["L4", "L40", "RTX 40"]):
            arch = "8.9"
        elif any(k in gpu_name for k in ["T4", "RTX 20", "GTX 16"]):
            arch = "7.5"
        elif any(k in gpu_name for k in ["V100"]):
            arch = "7.0"
        elif any(k in gpu_name for k in ["H100", "H200"]):
            arch = "9.0"

        os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        os.environ["CUDA_MODULE_LOADING"] = "LAZY"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        return arch
    except ImportError:
        return "cpu"


def detect_hardware_environment() -> Dict[str, Any]:
    """Phân tích phần cứng và đề xuất cấu hình huấn luyện tối ưu."""
    try:
        import torch
        is_cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if is_cuda else "CPU (No GPU)"
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if is_cuda else 0.0
    except ImportError:
        is_cuda = False
        gpu_name = "CPU (Torch not installed)"
        vram_gb = 0.0

    arch = setup_cuda_environment() if is_cuda else "None"

    # Khuyến nghị cấu hình dựa trên VRAM
    if vram_gb >= 35: # A100 (40GB/80GB)
        rec_fp8 = False
        rec_batch_size = 2
        rec_resolution = [1024, 1024]
        device_tier = "High-End (A100)"
    elif vram_gb >= 20: # L4 (24GB) / RTX 3090 / 4090
        rec_fp8 = True
        rec_batch_size = 1
        rec_resolution = [1024, 1024]
        device_tier = "Mid-End (L4/24GB)"
    else: # T4 (15GB/16GB) hoặc tương đương
        rec_fp8 = True
        rec_batch_size = 1
        rec_resolution = [768, 768]
        device_tier = "Standard / Free Tier (T4 16GB)"

    return {
        "is_cuda": is_cuda,
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 2),
        "cuda_arch": arch,
        "device_tier": device_tier,
        "recommended_fp8": rec_fp8,
        "recommended_batch_size": rec_batch_size,
        "recommended_resolution": rec_resolution,
    }
