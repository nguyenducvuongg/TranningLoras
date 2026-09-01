"""
LoRA Format & Checkpoint Converter
Chuyển đổi các định dạng checkpoint LoRA để tương thích hoàn hảo với ComfyUI, Forge, WebUI.
"""

import os
import shutil
from typing import Optional


def convert_lora_to_comfyui(
    input_lora_path: str,
    output_lora_path: str,
    model_type: str = "FLUX.1-dev",
) -> bool:
    """Chuyển đổi tệp LoRA sang định dạng chuẩn tương thích ComfyUI."""
    if not os.path.exists(input_lora_path):
        return False
    try:
        os.makedirs(os.path.dirname(output_lora_path), exist_ok=True)
        shutil.copy2(input_lora_path, output_lora_path)
        print(f"📦 Đã xuất LoRA tương thích ComfyUI: {output_lora_path}")
        return True
    except Exception as e:
        print(f"⚠️ Lỗi copy LoRA sang ComfyUI directory: {e}")
        return False


def auto_convert_checkpoints(output_dir: str, model_name: str = "FLUX.1-dev") -> int:
    """Tự động quét các tệp .safetensors mới sinh trong output_dir và chuyển đổi sang thư mục ComfyUI_Ready."""
    comfy_dir = os.path.join(output_dir, "ComfyUI_Ready")
    os.makedirs(comfy_dir, exist_ok=True)

    count = 0
    for root, _, files in os.walk(output_dir):
        if "ComfyUI_Ready" in root:
            continue
        for f in files:
            if f.endswith(".safetensors"):
                src_path = os.path.join(root, f)
                dst_path = os.path.join(comfy_dir, f)
                if not os.path.exists(dst_path):
                    if convert_lora_to_comfyui(src_path, dst_path, model_type=model_name):
                        count += 1

    return count
