"""
Model Registry
Chứa thông tin cấu hình, URL tải xuống chính thức từ HuggingFace và phân loại kiến trúc cho tất cả các mô hình hỗ trợ.
"""

from typing import Dict, Any, Optional

# Registry các VAE tiêu chuẩn
VAE_REGISTRY: Dict[str, str] = {
    "flux_vae": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/vae/diffusion_pytorch_model.safetensors",
    "flux2_vae": "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/vae/flux2-vae.safetensors",
    "qwen_vae": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_vae.safetensors",
    "z_image_vae": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
    "wan21_vae": "https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/resolve/main/Wan2.1_VAE.pth",
    "wan22_vae": "https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/resolve/main/Wan2.1_VAE.pth",
    "sdxl_vae": "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors",
}

# Registry các Text Encoder tiêu chuẩn
TEXT_ENCODER_REGISTRY: Dict[str, str] = {
    "clip_l": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
    "t5xxl_fp16": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors",
    "t5xxl_fp8": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
    "qwen_2_5_vl_7b": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors",
    "qwen_3_4b": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors",
    "qwen_3_8b": "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/text_encoders/qwen_3_8b.safetensors",
    "wan21_t5": "https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth",
    "wan21_clip_vision": "https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/resolve/main/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    "z_adapter_v2": "https://huggingface.co/ostris/zimage_turbo_training_adapter/resolve/main/zimage_turbo_training_adapter_v2.safetensors",
}

# Registry toàn diện các kiến trúc Model
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ================= FLUX ECOSYSTEM =================
    "FLUX.1-dev": {
        "engine": "toolkit",
        "type": "image",
        "arch": "flux",
        "name_or_path": "black-forest-labs/FLUX.1-dev",
        "vae": "flux_vae",
        "clip": "clip_l",
        "clip2": "t5xxl_fp16",
        "discrete_flow_shift": 3.0,
        "default_resolution": [1024, 1024],
        "supports_control": False,
    },
    "FLUX.1-schnell": {
        "engine": "toolkit",
        "type": "image",
        "arch": "flux",
        "name_or_path": "black-forest-labs/FLUX.1-schnell",
        "vae": "flux_vae",
        "clip": "clip_l",
        "clip2": "t5xxl_fp16",
        "discrete_flow_shift": 3.0,
        "default_resolution": [1024, 1024],
        "supports_control": False,
    },
    "FLUX.1-Kontext-dev": {
        "engine": "musubi",
        "alt_engine": "toolkit",
        "type": "image",
        "arch": "flux_kontext",
        "musubi_train_script": "src/musubi_tuner/flux_train_network.py",
        "download_url": "https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev/resolve/main/flux1-kontext-dev.safetensors",
        "vae": "flux_vae",
        "clip": "clip_l",
        "clip2": "t5xxl_fp16",
        "discrete_flow_shift": 3.0,
        "default_resolution": [1024, 1024],
        "supports_control": True,
    },
    "FLUX.2-klein-base-9B": {
        "engine": "musubi",
        "type": "image",
        "arch": "flux2",
        "musubi_train_script": "src/musubi_tuner/flux_2_train_network.py",
        "download_url": "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/diffusion_models/flux-2-klein-base-9b.safetensors",
        "vae": "flux2_vae",
        "clip": "qwen_3_8b",
        "model_version": "flux2_klein_9b",
        "discrete_flow_shift": 3.0,
        "default_resolution": [1024, 1024],
        "supports_control": True,
    },
    "FLUX.2-klein-base-4B": {
        "engine": "musubi",
        "type": "image",
        "arch": "flux2",
        "musubi_train_script": "src/musubi_tuner/flux_2_train_network.py",
        "download_url": "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b/resolve/main/split_files/diffusion_models/flux-2-klein-base-4b.safetensors",
        "vae": "flux2_vae",
        "clip": "qwen_3_4b",
        "model_version": "flux2_klein_4b",
        "discrete_flow_shift": 3.0,
        "default_resolution": [1024, 1024],
        "supports_control": True,
    },

    # ================= QWEN ECOSYSTEM =================
    "Qwen-Image": {
        "engine": "musubi",
        "type": "image",
        "arch": "qwen_image",
        "musubi_train_script": "src/musubi_tuner/qwen_image_train_network.py",
        "download_url": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_bf16.safetensors",
        "vae": "qwen_vae",
        "clip": "qwen_2_5_vl_7b",
        "model_version": "qwen_image",
        "discrete_flow_shift": 3.0,
        "default_resolution": [1024, 1024],
        "supports_control": False,
    },
    "Qwen-Image-Edit": {
        "engine": "musubi",
        "type": "image",
        "arch": "qwen_image_edit",
        "musubi_train_script": "src/musubi_tuner/qwen_image_train_network.py",
        "download_url": "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_bf16.safetensors",
        "vae": "qwen_vae",
        "clip": "qwen_2_5_vl_7b",
        "model_version": "qwen_image_edit",
        "discrete_flow_shift": 3.0,
        "default_resolution": [1024, 1024],
        "supports_control": True,
    },
    "Qwen-Image-Edit-2509": {
        "engine": "musubi",
        "type": "image",
        "arch": "qwen_image_edit",
        "musubi_train_script": "src/musubi_tuner/qwen_image_train_network.py",
        "download_url": "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2509_bf16.safetensors",
        "vae": "qwen_vae",
        "clip": "qwen_2_5_vl_7b",
        "model_version": "qwen_image_edit_2509",
        "discrete_flow_shift": 3.0,
        "default_resolution": [1024, 1024],
        "supports_control": True,
    },
    "Qwen-Image-Edit-2511": {
        "engine": "musubi",
        "type": "image",
        "arch": "qwen_image_edit",
        "musubi_train_script": "src/musubi_tuner/qwen_image_train_network.py",
        "download_url": "https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors",
        "vae": "qwen_vae",
        "clip": "qwen_2_5_vl_7b",
        "model_version": "qwen_image_edit_2511",
        "discrete_flow_shift": 3.0,
        "default_resolution": [1024, 1024],
        "supports_control": True,
    },

    # ================= Z-IMAGE ECOSYSTEM =================
    "Z-Image-Turbo": {
        "engine": "musubi",
        "type": "image",
        "arch": "z_image",
        "musubi_train_script": "src/musubi_tuner/zimage_train_network.py",
        "download_url": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors",
        "vae": "z_image_vae",
        "clip": "qwen_3_4b",
        "adapter": "z_adapter_v2",
        "discrete_flow_shift": 3.0,
        "default_resolution": [1024, 1024],
        "supports_control": False,
    },
    "Z-Image-Base": {
        "engine": "musubi",
        "type": "image",
        "arch": "z_image",
        "musubi_train_script": "src/musubi_tuner/zimage_train_network.py",
        "download_url": "https://huggingface.co/Comfy-Org/z_image/resolve/main/split_files/diffusion_models/z_image_bf16.safetensors",
        "vae": "z_image_vae",
        "clip": "qwen_3_4b",
        "discrete_flow_shift": 3.0,
        "default_resolution": [1024, 1024],
        "supports_control": False,
    },
    "Z-Image-De-Turbo": {
        "engine": "musubi",
        "type": "image",
        "arch": "z_image",
        "musubi_train_script": "src/musubi_tuner/zimage_train_network.py",
        "download_url": "https://huggingface.co/ostris/Z-Image-De-Turbo/resolve/main/z_image_de_turbo_v1_bf16.safetensors",
        "vae": "z_image_vae",
        "clip": "qwen_3_4b",
        "adapter": "z_adapter_v2",
        "discrete_flow_shift": 3.0,
        "default_resolution": [1024, 1024],
        "supports_control": False,
    },

    # ================= WAN 2.1 VIDEO ECOSYSTEM =================
    "Wan2.1-T2V-14B": {
        "engine": "musubi",
        "type": "video",
        "arch": "wan21",
        "musubi_train_script": "wan_train_network.py",
        "download_url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_14B_fp8_e4m3fn.safetensors",
        "vae": "wan21_vae",
        "clip": "wan21_t5",
        "discrete_flow_shift": 3.0,
        "default_resolution": [720, 1280],
        "supports_video": True,
        "supports_i2v": False,
    },
    "Wan2.1-I2V-14B-720P": {
        "engine": "musubi",
        "type": "video",
        "arch": "wan21",
        "musubi_train_script": "wan_train_network.py",
        "download_url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_720p_14B_fp8_e4m3fn.safetensors",
        "vae": "wan21_vae",
        "clip": "wan21_t5",
        "clip_vision": "wan21_clip_vision",
        "discrete_flow_shift": 3.0,
        "default_resolution": [720, 1280],
        "supports_video": True,
        "supports_i2v": True,
    },
    "Wan2.1-I2V-14B-480P": {
        "engine": "musubi",
        "type": "video",
        "arch": "wan21",
        "musubi_train_script": "wan_train_network.py",
        "download_url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors",
        "vae": "wan21_vae",
        "clip": "wan21_t5",
        "clip_vision": "wan21_clip_vision",
        "discrete_flow_shift": 3.0,
        "default_resolution": [480, 832],
        "supports_video": True,
        "supports_i2v": True,
    },
    "Wan2.1-T2V-1.3B": {
        "engine": "musubi",
        "type": "video",
        "arch": "wan21",
        "musubi_train_script": "wan_train_network.py",
        "download_url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors",
        "vae": "wan21_vae",
        "clip": "wan21_t5",
        "discrete_flow_shift": 3.0,
        "default_resolution": [480, 832],
        "supports_video": True,
        "supports_i2v": False,
    },

    # ================= WAN 2.2 VIDEO ECOSYSTEM =================
    "Wan2.2-T2V-14B": {
        "engine": "musubi",
        "type": "video",
        "arch": "wan22",
        "musubi_train_script": "wan_train_network.py",
        "download_url": "https://huggingface.co/Wan-AI/Wan2.2-T2V-14B/resolve/main/wan2.2_t2v_14B_fp8.safetensors",
        "vae": "wan22_vae",
        "clip": "wan21_t5",
        "discrete_flow_shift": 3.0,
        "default_boundary": 875,
        "default_resolution": [720, 1280],
        "supports_video": True,
        "supports_i2v": False,
    },
    "Wan2.2-I2V-14B": {
        "engine": "musubi",
        "type": "video",
        "arch": "wan22",
        "musubi_train_script": "wan_train_network.py",
        "download_url": "https://huggingface.co/Wan-AI/Wan2.2-I2V-14B/resolve/main/wan2.2_i2v_14B_fp8.safetensors",
        "vae": "wan22_vae",
        "clip": "wan21_t5",
        "clip_vision": "wan21_clip_vision",
        "discrete_flow_shift": 3.0,
        "default_boundary": 900,
        "default_resolution": [720, 1280],
        "supports_video": True,
        "supports_i2v": True,
    },

    # ================= KREA & OTHER =================
    "Krea2-Raw": {
        "engine": "musubi",
        "alt_engine": "toolkit",
        "type": "image",
        "arch": "krea2",
        "musubi_train_script": "src/musubi_tuner/krea2_train_network.py",
        "download_url": "https://huggingface.co/krea/Krea-2-Raw/resolve/main/raw.safetensors",
        "vae": "flux_vae",
        "clip": "clip_l",
        "clip2": "t5xxl_fp16",
        "discrete_flow_shift": 3.0,
        "default_resolution": [1024, 1024],
        "supports_control": False,
    }
}


def _clean_key(k: str) -> str:
    """Loại bỏ ký tự đặc biệt để so sánh fuzzy."""
    return k.lower().replace(" ", "").replace("_", "").replace("-", "").replace(".", "")


def get_model_info(model_key: str) -> Dict[str, Any]:
    """Lấy metadata của model từ registry, hỗ trợ alias linh hoạt."""
    normalized_key = model_key.strip()
    if normalized_key in MODEL_REGISTRY:
        return MODEL_REGISTRY[normalized_key]
    
    # Fuzzy match
    cleaned_input = _clean_key(normalized_key)
    for k, v in MODEL_REGISTRY.items():
        if _clean_key(k) == cleaned_input:
            return v
            
    raise ValueError(f"Mô hình '{model_key}' không tồn tại trong danh mục MODEL_REGISTRY. Các model hỗ trợ: {list(MODEL_REGISTRY.keys())}")


def get_preferred_engine(model_key: str) -> str:
    """Trả về engine huấn luyện tối ưu nhất (musubi hoặc toolkit) cho model."""
    info = get_model_info(model_key)
    return info.get("engine", "musubi")
