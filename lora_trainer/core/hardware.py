"""
Hardware & GPU Detection Engine
Nhận diện môi trường phần cứng, GPU, VRAM và đưa ra khuyến nghị siêu tham số tối ưu cho từng nền tảng GPU.
"""

import os
import subprocess
from typing import Dict, Any


def detect_hardware_environment() -> Dict[str, Any]:
    """
    Quét thông tin GPU và tài nguyên hệ thống.
    Trả về cấu hình gợi ý tối ưu theo từng dòng GPU (T4, L4, A100, V100, v.v.).
    """
    gpu_name = "Unknown GPU / CPU Only"
    vram_gb = 0.0
    device_tier = "cpu"

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = round(vram_bytes / (1024 ** 3), 2)
    except Exception:
        pass

    if vram_gb == 0.0:
        try:
            cmd = "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits"
            output = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
            if output:
                parts = output.split(",")
                gpu_name = parts[0].strip()
                vram_gb = round(float(parts[1].strip()) / 1024.0, 2)
        except Exception:
            pass

    # Phân loại GPU Tier
    if vram_gb >= 35.0:
        device_tier = "enterprise_vram" # A100, H100
        recommended_batch_size = 2
        recommended_fp8 = False
        recommended_precision = "bf16"
        recommended_optimizer = "adamw8bit"
        recommended_resolution = [1024, 1024]
        recommended_caching = True
    elif vram_gb >= 20.0:
        device_tier = "high_vram" # L4 (24GB), RTX 3090/4090, V100 32GB
        recommended_batch_size = 1
        recommended_fp8 = True
        recommended_precision = "bf16"
        recommended_optimizer = "adamw8bit"
        recommended_resolution = [1024, 1024]
        recommended_caching = True
    elif vram_gb >= 14.0:
        device_tier = "medium_vram" # T4 (15-16GB), V100 16GB
        recommended_batch_size = 1
        recommended_fp8 = True
        recommended_precision = "fp16"
        recommended_optimizer = "adamw8bit"
        recommended_resolution = [1024, 1024]
        recommended_caching = True
    else:
        device_tier = "low_vram" # CPU / <14GB
        recommended_batch_size = 1
        recommended_fp8 = True
        recommended_precision = "fp16"
        recommended_optimizer = "adamw8bit"
        recommended_resolution = [512, 512]
        recommended_caching = True

    return {
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "device_tier": device_tier,
        "recommended_fp8": recommended_fp8,
        "recommended_precision": recommended_precision,
        "recommended_batch_size": recommended_batch_size,
        "recommended_optimizer": recommended_optimizer,
        "recommended_resolution": recommended_resolution,
        "recommended_caching": recommended_caching,
    }


def setup_cuda_environment() -> None:
    """Thiết lập các biến môi trường PyTorch / CUDA chống tràn bộ nhớ phân mảnh."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
