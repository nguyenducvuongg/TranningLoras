"""
Universal Model Registry
Chứa thông tin cấu hình, URL tải xuống chính thức (100% Direct Public Mirrors)
và phân loại kiến trúc cho tất cả các mô hình: SDXL (Pony, Illustrious), SD1.5, SD3.5,
FLUX.1, FLUX.2 Klein, Qwen-Image, Z-Image, Krea2, Wan 2.1/2.2 Video và Custom Models.
"""

from typing import Dict, Any, List, Optional

# Registry các VAE tiêu chuẩn (100% Public Direct Mirrors - 200 OK)
VAE_REGISTRY: Dict[str, str] = {
    "flux_vae": "https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/ae.safetensors",
    "flux2_vae": "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/vae/flux2-vae.safetensors",
    "qwen_vae": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors",
    "qwen_image_vae": "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors",
    "z_image_vae": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
    "sdxl_vae": "https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors",
    "sd15_vae": "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors",
}

VAE_FALLBACKS: Dict[str, str] = {
    "flux_vae": "https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/ae.safetensors",
    "flux2_vae": "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/vae/flux2-vae.safetensors",
    "qwen_vae": "https://huggingface.co/StableDiffusionVN/QwenImage/resolve/main/vae/qwen_vae.safetensors",
    "qwen_image_vae": "https://huggingface.co/StableDiffusionVN/QwenImage/resolve/main/vae/qwen_vae.safetensors",
    "sdxl_vae": "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors",
}

# Registry các Text Encoder tiêu chuẩn (100% Public Direct Mirrors - 200 OK)
TEXT_ENCODER_REGISTRY: Dict[str, str] = {
    "clip_l": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
    "t5xxl_fp16": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors",
    "t5xxl_fp8": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
    "qwen_2_5_vl_7b": "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors",
    "qwen_3_4b": "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors",
    "qwen_3_8b": "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/text_encoders/qwen_3_8b.safetensors",
    "qwen3vl_4b": "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_bf16.safetensors",
    "z_adapter_v2": "https://huggingface.co/ostris/zimage_turbo_training_adapter/resolve/main/zimage_turbo_training_adapter_v2.safetensors",
}

TEXT_ENCODER_FALLBACKS: Dict[str, str] = {
    "clip_l": "https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/clip_l.safetensors",
    "t5xxl_fp16": "https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/t5xxl_fp16.safetensors",
    "t5xxl_fp8": "https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
    "qwen3vl_4b": "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_bf16.safetensors",
}

# Registry toàn diện các kiến trúc Model
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ================= SDXL & PONY / ILLUSTRIOUS ECOSYSTEM =================
    "SDXL-Base-1.0": {
        "engine": "sdscripts",
        "alt_engine": "toolkit",
        "category": "sdxl",
        "type": "image",
        "arch": "sdxl",
        "download_url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
        "fallback_url": "https://huggingface.co/bdsqlsz/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
        "vae": "sdxl_vae",
        "default_resolution": [1024, 1024],
        "supports_control": False,
    },
    "Pony-Diffusion-V6-XL": {
        "engine": "sdscripts",
        "alt_engine": "toolkit",
        "category": "sdxl",
        "type": "image",
        "arch": "sdxl",
        "download_url": "https://huggingface.co/AstraliteHeart/pony-diffusion-v6-xl/resolve/main/v6_pruned.safetensors",
        "fallback_url": "https://civitai.com/api/download/models/290640",
        "vae": "sdxl_vae",
        "default_resolution": [1024, 1024],
        "supports_control": False,
    },
    "Illustrious-XL-v0.1": {
        "engine": "sdscripts",
        "alt_engine": "toolkit",
        "category": "sdxl",
        "type": "image",
        "arch": "sdxl",
        "download_url": "https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0.1/resolve/main/Illustrious-XL-v0.1.safetensors",
        "vae": "sdxl_vae",
        "default_resolution": [1024, 1024],
        "supports_control": False,
    },
    "Animagine-XL-3.1": {
        "engine": "sdscripts",
        "alt_engine": "toolkit",
        "category": "sdxl",
        "type": "image",
        "arch": "sdxl",
        "download_url": "https://huggingface.co/cagliostrolab/animagine-xl-3.1/resolve/main/animagine-xl-3.1.safetensors",
        "vae": "sdxl_vae",
        "default_resolution": [1024, 1024],
        "supports_control": False,
    },
    "DreamShaper-XL": {
        "engine": "sdscripts",
        "alt_engine": "toolkit",
        "category": "sdxl",
        "type": "image",
        "arch": "sdxl",
        "download_url": "https://huggingface.co/Lykon/dreamshaper-xl-v2-turbo/resolve/main/DreamShaperXL_Turbo_v2_1.safetensors",
        "vae": "sdxl_vae",
        "default_resolution": [1024, 1024],
        "supports_control": False,
    },

    # ================= STABLE DIFFUSION 1.5 ECOSYSTEM =================
    "v1-5-pruned-emaonly": {
        "engine": "sdscripts",
        "category": "sd15",
        "type": "image",
        "arch": "sd15",
        "download_url": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
        "vae": "sd15_vae",
        "default_resolution": [512, 512],
        "supports_control": False,
    },
    "Realistic-Vision-v5.1": {
        "engine": "sdscripts",
        "category": "sd15",
        "type": "image",
        "arch": "sd15",
        "download_url": "https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1_fp16-no-vae.safetensors",
        "vae": "sd15_vae",
        "default_resolution": [512, 512],
        "supports_control": False,
    },
    "DreamShaper-8": {
        "engine": "sdscripts",
        "category": "sd15",
        "type": "image",
        "arch": "sd15",
        "download_url": "https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors",
        "vae": "sd15_vae",
        "default_resolution": [512, 512],
        "supports_control": False,
    },

    # ================= STABLE DIFFUSION 3 / 3.5 =================
    "SD3.5-Large": {
        "engine": "sdscripts",
        "category": "sd35",
        "type": "image",
        "arch": "sd35",
        "download_url": "https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/sd3.5_large.safetensors",
        "fallback_url": "https://huggingface.co/Comfy-Org/stable-diffusion-3.5-large/resolve/main/sd3.5_large.safetensors",
        "clip": "clip_l",
        "clip2": "t5xxl_fp16",
        "default_resolution": [1024, 1024],
        "supports_control": False,
    },
    "SD3.5-Large-Turbo": {
        "engine": "sdscripts",
        "category": "sd35",
        "type": "image",
        "arch": "sd35",
        "download_url": "https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo/resolve/main/sd3.5_large_turbo.safetensors",
        "clip": "clip_l",
        "clip2": "t5xxl_fp16",
        "default_resolution": [1024, 1024],
        "supports_control": False,
    },
    "SD3.5-Medium": {
        "engine": "sdscripts",
        "category": "sd35",
        "type": "image",
        "arch": "sd35",
        "download_url": "https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/resolve/main/sd3.5_medium.safetensors",
        "clip": "clip_l",
        "clip2": "t5xxl_fp16",
        "default_resolution": [1024, 1024],
        "supports_control": False,
    },

    # ================= FLUX.1 ECOSYSTEM =================
    "FLUX.1-dev": {
        "engine": "toolkit",
        "alt_engine": "musubi",
        "category": "flux",
        "type": "image",
        "arch": "flux",
        "toolkit_arch": "flux",
        "name_or_path": "black-forest-labs/FLUX.1-dev",
        "download_url": "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors",
        "fallback_url": "https://huggingface.co/camenduru/FLUX.1-dev/resolve/main/flux1-dev.safetensors",
        "vae": "flux_vae",
        "clip": "clip_l",
        "clip2": "t5xxl_fp16",
        "discrete_flow_shift": 3.0,
        "default_resolution": [1024, 1024],
        "supports_control": False,
    },
    "FLUX.1-schnell": {
        "engine": "toolkit",
        "alt_engine": "musubi",
        "category": "flux",
        "type": "image",
        "arch": "flux",
        "toolkit_arch": "flux",
        "name_or_path": "black-forest-labs/FLUX.1-schnell",
        "download_url": "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors",
        "fallback_url": "https://huggingface.co/camenduru/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors",
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
        "category": "flux",
        "type": "image",
        "arch": "flux_kontext",
        "toolkit_arch": "flux_kontext",
        "musubi_train_script": "src/musubi_tuner/flux_kontext_train_network.py",
        "download_url": "https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors",
        "fallback_url": "https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors",
        "vae": "flux_vae",
        "clip": "clip_l",
        "clip2": "t5xxl_fp16",
        "discrete_flow_shift": 3.0,
        "default_resolution": [1024, 1024],
        "supports_control": True,
    },
    "FLUX.1-Fill-dev": {
        "engine": "toolkit",
        "category": "flux",
        "type": "image",
        "arch": "flux_fill",
        "toolkit_arch": "flux_fill",
        "name_or_path": "black-forest-labs/FLUX.1-Fill-dev",
        "vae": "flux_vae",
        "clip": "clip_l",
        "clip2": "t5xxl_fp16",
        "discrete_flow_shift": 3.0,
        "default_resolution": [1024, 1024],
        "supports_control": True,
    },

    # ================= FLUX.2 / KLEIN ECOSYSTEM =================
    "FLUX.2-klein-base-9B": {
        "engine": "musubi",
        "alt_engine": "toolkit",
        "category": "flux2",
        "type": "image",
        "arch": "flux2",
        "toolkit_arch": "flux2_klein_9b",
        "musubi_train_script": "src/musubi_tuner/flux_2_train_network.py",
        "download_url": "https://huggingface.co/zhangchenxu/FLUX.2-klein-base-9B/resolve/main/flux-2-klein-base-9b.safetensors",
        "fallback_url": "https://huggingface.co/SassyDiffusion/FLUX.2-klein-base-9B-bf16/resolve/main/flux-2-klein-base-9b.safetensors",
        "vae": "flux2_vae",
        "clip": "qwen_3_8b",
        "model_version": "flux2_klein_9b",
        "discrete_flow_shift": 3.0,
        "default_resolution": [1024, 1024],
        "supports_control": True,
    },
    "FLUX.2-klein-base-4B": {
        "engine": "musubi",
        "alt_engine": "toolkit",
        "category": "flux2",
        "type": "image",
        "arch": "flux2",
        "toolkit_arch": "flux2_klein_4b",
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
        "alt_engine": "toolkit",
        "category": "qwen",
        "type": "image",
        "arch": "qwen_image",
        "toolkit_arch": "qwen_image:2512",
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
        "alt_engine": "toolkit",
        "category": "qwen",
        "type": "image",
        "arch": "qwen_image_edit",
        "toolkit_arch": "qwen_image_edit_plus",
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
        "alt_engine": "toolkit",
        "category": "qwen",
        "type": "image",
        "arch": "qwen_image_edit",
        "toolkit_arch": "qwen_image_edit_plus",
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
        "alt_engine": "toolkit",
        "category": "qwen",
        "type": "image",
        "arch": "qwen_image_edit",
        "toolkit_arch": "qwen_image_edit_plus:2511",
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
        "category": "z_image",
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
        "category": "z_image",
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
        "category": "z_image",
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

    # ================= KREA & DIT ECOSYSTEM =================
    "Krea2-Raw": {
        "engine": "musubi",
        "alt_engine": "toolkit",
        "category": "krea",
        "type": "image",
        "arch": "krea2",
        "toolkit_arch": "krea2",
        "musubi_train_script": "src/musubi_tuner/krea2_train_network.py",
        "download_url": "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_raw_bf16.safetensors",
        "fallback_url": "https://huggingface.co/krea/Krea-2-Raw/resolve/main/raw.safetensors",
        "vae": "qwen_image_vae",
        "clip": "qwen3vl_4b",
        "discrete_flow_shift": 2.5,
        "default_resolution": [1024, 1024],
        "supports_control": False,
    },
    "Sana-1.6B": {
        "engine": "toolkit",
        "category": "sana",
        "type": "image",
        "arch": "sana",
        "toolkit_arch": "sana",
        "name_or_path": "Efficient-Large-Model/Sana_1600M_1024px",
        "download_url": "https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px/resolve/main/checkpoints/Sana_1600M_1024px.pth",
        "default_resolution": [1024, 1024],
        "supports_control": False,
    },

    # ================= CUSTOM / GENERIC MODELS =================
    "Custom-SDXL": {
        "engine": "sdscripts",
        "alt_engine": "toolkit",
        "category": "sdxl",
        "type": "image",
        "arch": "sdxl",
        "vae": "sdxl_vae",
        "default_resolution": [1024, 1024],
        "supports_control": False,
    },
    "Custom-SD15": {
        "engine": "sdscripts",
        "category": "sd15",
        "type": "image",
        "arch": "sd15",
        "vae": "sd15_vae",
        "default_resolution": [512, 512],
        "supports_control": False,
    },
    "Custom-FLUX": {
        "engine": "toolkit",
        "alt_engine": "musubi",
        "category": "flux",
        "type": "image",
        "arch": "flux",
        "toolkit_arch": "flux",
        "vae": "flux_vae",
        "clip": "clip_l",
        "clip2": "t5xxl_fp16",
        "discrete_flow_shift": 3.0,
        "default_resolution": [1024, 1024],
        "supports_control": False,
    },
}


def _clean_key(k: str) -> str:
    """Loại bỏ ký tự đặc biệt để so sánh fuzzy."""
    return k.lower().replace(" ", "").replace("_", "").replace("-", "").replace(".", "")


def get_model_info(model_key: str, custom_download_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Lấy metadata của model từ registry, hỗ trợ alias linh hoạt và tự động sinh metadata cho Custom Model.
    """
    normalized_key = model_key.strip()
    if normalized_key in MODEL_REGISTRY:
        info = dict(MODEL_REGISTRY[normalized_key])
        if custom_download_url:
            info["download_url"] = custom_download_url
        return info

    # Fuzzy match
    cleaned_input = _clean_key(normalized_key)
    for k, v in MODEL_REGISTRY.items():
        if _clean_key(k) == cleaned_input:
            info = dict(v)
            if custom_download_url:
                info["download_url"] = custom_download_url
            return info

    # Xử lý nếu là Custom URL hoặc Checkpoint đường dẫn cục bộ
    if custom_download_url or normalized_key.startswith("http://") or normalized_key.startswith("https://") or normalized_key.endswith(".safetensors"):
        url = custom_download_url or normalized_key
        # Mặc định suy luận kiến trúc SDXL nếu không có thông tin
        return {
            "engine": "sdscripts",
            "category": "custom",
            "type": "image",
            "arch": "sdxl",
            "download_url": url,
            "vae": "sdxl_vae",
            "default_resolution": [1024, 1024],
            "supports_control": False,
        }

    raise ValueError(
        f"Mô hình '{model_key}' không tồn tại trong danh mục MODEL_REGISTRY. "
        f"Các model hỗ trợ: {list(MODEL_REGISTRY.keys())}"
    )


def get_preferred_engine(model_key: str) -> str:
    """Trả về engine huấn luyện tối ưu nhất (sdscripts, musubi, hoặc toolkit) cho model."""
    info = get_model_info(model_key)
    return info.get("engine", "musubi")


def list_supported_models(category: Optional[str] = None) -> List[str]:
    """Trả về danh sách tên mô hình được hỗ trợ, có thể lọc theo phân loại."""
    if category is None:
        return list(MODEL_REGISTRY.keys())
    cat_clean = category.lower().strip()
    return [k for k, v in MODEL_REGISTRY.items() if v.get("category", "").lower() == cat_clean or v.get("type", "").lower() == cat_clean]
