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

    # Phân loại GPU Tier & Siêu tham số tối ưu (Chống OOM & Chống Da Nhựa)
    if vram_gb >= 35.0:
        device_tier = "enterprise_vram" # A100, H100
        recommended_batch_size = 2
        recommended_fp8 = False
        recommended_precision = "bf16"
        recommended_optimizer = "adamw8bit"
        recommended_resolution = [1024, 1024]
        recommended_caching = True
        recommended_noise_offset = 0.06
        recommended_min_snr_gamma = 5
    elif vram_gb >= 20.0:
        device_tier = "high_vram" # L4 (24GB), RTX 3090/4090, V100 32GB
        recommended_batch_size = 1
        recommended_fp8 = True
        recommended_precision = "bf16"
        recommended_optimizer = "adamw8bit"
        recommended_resolution = [1024, 1024]
        recommended_caching = True
        recommended_noise_offset = 0.06
        recommended_min_snr_gamma = 5
    elif vram_gb >= 14.0:
        device_tier = "medium_vram" # T4 (15-16GB), V100 16GB
        recommended_batch_size = 1
        recommended_fp8 = True
        recommended_precision = "fp16"
        recommended_optimizer = "adamw8bit"
        recommended_resolution = [1024, 1024]
        recommended_caching = True
        recommended_noise_offset = 0.06
        recommended_min_snr_gamma = 5
    else:
        device_tier = "low_vram" # CPU / <14GB
        recommended_batch_size = 1
        recommended_fp8 = True
        recommended_precision = "fp16"
        recommended_optimizer = "adamw8bit"
        recommended_resolution = [512, 512]
        recommended_caching = True
        recommended_noise_offset = 0.05
        recommended_min_snr_gamma = 5

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
        "recommended_noise_offset": recommended_noise_offset,
        "recommended_min_snr_gamma": recommended_min_snr_gamma,
    }


def generate_accelerate_config(
    output_path: str = "/root/.cache/huggingface/accelerate/default_config.yaml",
    precision: str = "fp16",
    num_processes: int = 1,
    sync_to_drive_dir: str = "/content/drive/MyDrive/TranningLorasData/config",
) -> str:
    """
    Sinh tệp cấu hình Accelerate chuẩn hóa tối ưu cho Single-GPU:
    - Khóa num_processes = 1 chống đa tiến trình nhầm lẫn
    - Đồng bộ mixed_precision tương thích phần cứng (fp16 cho T4, bf16 cho L4/A100)
    - Lưu vào default_config.yaml và sao lưu vào Google Drive.
    """
    config_content = f"""compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: '0'
machine_rank: 0
main_training_function: main
mixed_precision: {precision}
num_machines: 1
num_processes: {num_processes}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""
    # 1. Ghi vào cache hệ thống
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(config_content)
    except Exception as e:
        print(f"⚠️ Không thể tạo {output_path}: {e}")

    # 2. Đồng bộ vào Google Drive nếu thư mục tồn tại
    if sync_to_drive_dir and os.path.exists(sync_to_drive_dir):
        try:
            drive_yaml = os.path.join(sync_to_drive_dir, "accelerate_config.yaml")
            with open(drive_yaml, "w", encoding="utf-8") as f:
                f.write(config_content)
            print(f"⚡ Đã đồng bộ cấu hình Accelerate ({precision}) vào Google Drive: {drive_yaml}")
        except Exception:
            pass

    return output_path


def setup_cuda_environment() -> None:
    """Thiết lập các biến môi trường PyTorch / CUDA chống tràn bộ nhớ phân mảnh và sinh config Accelerate."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONUNBUFFERED"] = "1"

    hw = detect_hardware_environment()
    try:
        generate_accelerate_config(precision=hw.get("recommended_precision", "fp16"))
    except Exception:
        pass
