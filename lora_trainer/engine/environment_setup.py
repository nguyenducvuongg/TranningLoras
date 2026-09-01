"""
Environment Setup & Accelerate Configuration Manager
Tự động cấu hình môi trường thực thi tối ưu cho GPU, sinh tệp Accelerate Config mặc định
và thiết lập các cờ tối ưu hóa bộ nhớ VRAM.
"""

import os
import sys
import yaml
from typing import Dict, Any, Optional

DEFAULT_ACCELERATE_CONFIG_PATH = os.path.expanduser("~/.cache/huggingface/accelerate/default_config.yaml")
BACKUP_ACCELERATE_CONFIG_PATH = "/content/drive/MyDrive/TranningLorasData/config/accelerate_config.yaml"


def setup_accelerate_config(
    custom_path: Optional[str] = None,
    mixed_precision: str = "bf16",
    num_processes: int = 1,
) -> str:
    """
    Tự động sinh file cấu hình Accelerate chuẩn hóa tối ưu cho Colab GPU (L4, A100, T4, V100).
    Lưu vào ~/.cache/huggingface/accelerate/default_config.yaml và sao lưu vào Google Drive.
    """
    config_data: Dict[str, Any] = {
        "compute_environment": "LOCAL_MACHINE",
        "debug": False,
        "distributed_type": "NO",
        "downcast_bf16": "no",
        "gpu_ids": "all",
        "machine_rank": 0,
        "main_training_function": "main",
        "mixed_precision": mixed_precision,
        "num_machines": 1,
        "num_processes": num_processes,
        "rdzv_backend": "static",
        "same_network": True,
        "tpu_env": [],
        "tpu_use_cluster": False,
        "tpu_use_sudo": False,
        "use_cpu": False,
    }

    target_path = custom_path or DEFAULT_ACCELERATE_CONFIG_PATH
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    with open(target_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, default_flow_style=False)

    # Sao lưu sang Google Drive nếu có kết nối Drive
    if os.path.exists("/content/drive/MyDrive"):
        try:
            os.makedirs(os.path.dirname(BACKUP_ACCELERATE_CONFIG_PATH), exist_ok=True)
            with open(BACKUP_ACCELERATE_CONFIG_PATH, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f, default_flow_style=False)
        except Exception:
            pass

    return target_path


def apply_performance_environment_vars(
    musubi_dir: str = "/content/musubi-tuner",
    toolkit_dir: str = "/content/ai-toolkit",
) -> None:
    """
    Thiết lập các biến môi trường hiệu năng cao cho PyTorch và CUDA.
    Chống phân mảnh bộ nhớ VRAM và hỗ trợ module path.
    """
    # Chống phân mảnh bộ nhớ VRAM / OOM
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    # Chèn đường dẫn module cho cả Musubi và AI-Toolkit
    existing_pp = os.environ.get("PYTHONPATH", "")
    src_musubi = os.path.join(musubi_dir, "src")
    paths_to_add = [src_musubi, musubi_dir, toolkit_dir]
    
    current_paths = existing_pp.split(":") if existing_pp else []
    for p in paths_to_add:
        if p not in current_paths:
            current_paths.insert(0, p)
            if p not in sys.path:
                sys.path.insert(0, p)

    os.environ["PYTHONPATH"] = ":".join(current_paths).strip(":")


def initialize_training_environment(
    base_drive_dir: str = "/content/drive/MyDrive/TranningLorasData",
) -> Dict[str, Any]:
    """
    Hàm tổng hợp thiết lập toàn bộ môi trường huấn luyện:
    1. Cấu hình Accelerate config
    2. Kích hoạt biến môi trường CUDA & VRAM
    3. Trả về thông tin môi trường
    """
    apply_performance_environment_vars()
    accel_cfg = setup_accelerate_config()

    return {
        "accelerate_config": accel_cfg,
        "cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "pythonpath": os.environ.get("PYTHONPATH"),
    }
