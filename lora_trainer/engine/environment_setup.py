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


ALL_TRAINER_DEPENDENCIES = [
    "torchvision",
    "transformers>=4.44.0",
    "diffusers>=0.30.0",
    "accelerate>=0.33.0",
    "safetensors>=0.4.4",
    "bitsandbytes>=0.43.0",
    "optimum-quanto>=0.2.4",
    "albumentations>=1.4.0",
    "opencv-python-headless>=4.8.0",
    "pyyaml>=6.0",
    "toml>=0.10.2",
    "pillow>=10.0.0",
    "tqdm>=4.66.0",
    "scipy>=1.11.0",
    "wandb>=0.16.0",
    "tensorboard",
    "matplotlib",
    "einops>=0.7.0",
    "imagesize>=1.4.1",
    "ftfy>=6.1.1",
    "regex>=2023.10.3",
    "sentencepiece>=0.1.99",
    "protobuf>=3.20.0",
    "av>=11.0.0",
    "imageio>=2.30.0",
    "imageio-ffmpeg>=0.4.8",
    "kornia>=0.7.0",
    "open_clip_torch>=2.24.0",
    "timm>=0.9.0",
    "prodigyopt>=1.0.0",
    "lion-pytorch>=0.1.2",
    "voluptuous>=0.13.0",
    "huggingface_hub>=0.24.0",
    "flatten_json",
    "pydantic",
    "clean-fid",
    "invisible-watermark",
    "google-genai",
    "openai",
    "ipywidgets",
]


def install_all_trainer_dependencies(quiet: bool = True) -> bool:
    """Tự động kiểm tra và cài đặt toàn bộ phụ thuộc cho cả Musubi-Tuner và AI-Toolkit."""
    import subprocess
    cmd = [sys.executable, "-m", "pip", "install"]
    if quiet:
        cmd.append("-q")
    cmd.extend(ALL_TRAINER_DEPENDENCIES)
    try:
        subprocess.run(cmd, check=False)
        return True
    except Exception:
        return False


def initialize_training_environment(
    base_drive_dir: str = "/content/drive/MyDrive/TranningLorasData",
    auto_install_deps: bool = False,
) -> Dict[str, Any]:
    """
    Hàm tổng hợp thiết lập toàn bộ môi trường huấn luyện:
    1. Cài đặt đầy đủ dependencies nếu yêu cầu
    2. Cấu hình Accelerate config
    3. Kích hoạt biến môi trường CUDA & VRAM
    4. Trả về thông tin môi trường
    """
    if auto_install_deps:
        install_all_trainer_dependencies()

    apply_performance_environment_vars()
    accel_cfg = setup_accelerate_config()

    return {
        "accelerate_config": accel_cfg,
        "cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "pythonpath": os.environ.get("PYTHONPATH"),
    }
