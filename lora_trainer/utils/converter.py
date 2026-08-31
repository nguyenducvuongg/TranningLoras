"""
LoRA Format Converter
Chuyển đổi các định dạng LoRA đặc thù (Z-LoRA, Diffusers) sang định dạng tương thích trực tiếp với ComfyUI & WebUI.
"""

import os
from typing import Optional


def convert_z_lora_to_comfyui(input_lora_path: str, output_lora_path: str) -> bool:
    """
    Chuyển đổi Z-Image LoRA sang định dạng ComfyUI chuẩn.
    Thực hiện mapping các key layer phù hợp với ComfyUI loader.
    """
    if not os.path.exists(input_lora_path):
        raise FileNotFoundError(f"Không tìm thấy file LoRA nguồn: {input_lora_path}")

    from safetensors.torch import load_file, save_file
    print(f"🔄 Đang chuyển đổi Z-LoRA sang ComfyUI: {input_lora_path} -> {output_lora_path}")
    state_dict = load_file(input_lora_path)
    new_state_dict = {}

    for k, v in state_dict.items():
        new_key = k
        # Xử lý mapping key theo chuẩn ComfyUI cho Z-Image
        if "lora_unet_" in k:
            new_key = k.replace("lora_unet_", "diffusion_model.")
        elif "lora_te_" in k:
            new_key = k.replace("lora_te_", "text_encoder.")
            
        new_state_dict[new_key] = v

    os.makedirs(os.path.dirname(output_lora_path), exist_ok=True)
    save_file(new_state_dict, output_lora_path)
    print(f"✅ Chuyển đổi thành công! Đã lưu tại: {output_lora_path}")
    return True


def convert_diffusers_to_safetensors(input_path: str, output_path: str) -> bool:
    """Chuyển đổi trọng số LoRA từ diffusers format sang single safetensors file."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Không tìm thấy: {input_path}")

    import torch
    from safetensors.torch import load_file, save_file

    if os.path.isdir(input_path):
        # Diffusers folder format
        bin_or_safe = os.path.join(input_path, "pytorch_lora_weights.safetensors")
        if not os.path.exists(bin_or_safe):
            bin_or_safe = os.path.join(input_path, "pytorch_lora_weights.bin")
            if not os.path.exists(bin_or_safe):
                raise FileNotFoundError("Không tìm thấy file weights trong thư mục diffusers!")
            state_dict = torch.load(bin_or_safe, map_location="cpu")
        else:
            state_dict = load_file(bin_or_safe)
    else:
        state_dict = load_file(input_path) if input_path.endswith(".safetensors") else torch.load(input_path, map_location="cpu")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_file(state_dict, output_path)
    print(f"✅ Đã lưu file safetensors tại: {output_path}")
    return True
