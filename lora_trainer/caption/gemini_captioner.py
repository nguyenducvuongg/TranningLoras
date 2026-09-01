"""
Gemini Multimodal AI Captioner
Tận dụng sức mạnh của toàn bộ dòng Google Gemini (2.5 Pro, 2.5 Flash, 2.5 Flash Lite, 2.0 Flash, 2.0 Pro Exp, 1.5 Pro, 1.5 Flash)
để sinh caption chất lượng cao chuyên sâu theo từng dạng LoRA (Da, Upscale, Phong cách, Nhân vật, Video).
"""

import os
import io
import time
from typing import Optional, List
from PIL import Image
from tqdm import tqdm
from .key_manager import get_api_key
from ..data.cleaner import get_supported_images, get_supported_videos

# Toàn bộ danh sách model Gemini chính thức qua API (Thế hệ Gemini 3.x)
GEMINI_MODELS = {
    "Gemini-3.6-Flash": "gemini-3.6-flash",
    "Gemini-3.7-Flash": "gemini-3.7-flash",
    "Gemini-3.5-Flash": "gemini-3.5-flash",
    "Gemini-3.5-Flash-Lite": "gemini-3.5-flash-lite",
    "Gemini-3.1-Pro": "gemini-3.1-pro-preview",
    "Gemini-3-Pro": "gemini-3-pro-preview",
    # Tương thích ngược và tự động chuyển hướng các model cũ đã deprecated
    "Gemini-2.0-Flash": "gemini-3.6-flash",
    "Gemini-2.0-Flash-Lite": "gemini-3.5-flash-lite",
    "Gemini-2.5-Flash": "gemini-3.6-flash",
    "Gemini-2.5-Pro": "gemini-3.1-pro-preview",
    "Gemini-1.5-Pro": "gemini-3.1-pro-preview",
    "Gemini-1.5-Flash": "gemini-3.6-flash",
    "gemini-2.0-flash": "gemini-3.6-flash",
    "gemini-2.5-flash": "gemini-3.6-flash",
    "gemini-2.5-pro": "gemini-3.1-pro-preview",
    "gemini-1.5-pro": "gemini-3.1-pro-preview",
    "gemini-1.5-flash": "gemini-3.6-flash",
}

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"

# Hướng dẫn độ dài caption
LENGTH_PROMPTS = {
    "Short": "Write a concise single-sentence caption, around 20-30 words.",
    "Medium": "Write a detailed, informative caption, around 50-70 words.",
    "Long": "Write a comprehensive, extremely detailed caption, around 100-150 words.",
}

# Các chế độ Prompt chuyên sâu theo từng loại LoRA
TASK_SYSTEM_PROMPTS = {
    "General": (
        "You are an expert AI dataset captioner for Diffusion model training. "
        "Describe the visual content objectively and clearly. Focus on main subject, lighting, action, and background. "
    ),
    "Skin_Portrait": (
        "You are a specialized AI portrait and beauty retouching dataset captioner. "
        "Focus on skin details: skin tone, natural pores, texture, subtle blemishes, makeup, facial features, eye reflections, and facial lighting. "
        "Do NOT use vague words like 'perfect face'; describe the real photographic skin textures and lighting realistically. "
    ),
    "Upscale_Restoration": (
        "You are an expert image super-resolution and enhancement dataset captioner. "
        "Describe the ultra-sharp micro-details, fine fabric weaves, hair strands, crisp edges, surface textures, and photographic clarity. "
    ),
    "Art_Style": (
        "You are an art style and aesthetic dataset captioner. "
        "Describe the subject matter, composition, and lighting objectively while omitting specific art style labels (so the model can bind the visual style to the trigger word). "
    ),
    "Character_Outfit": (
        "You are a character and fashion dataset captioner. "
        "Describe the character's exact pose, hairstyle, facial expression, clothing items, patterns, fabrics, and accessories with high precision. "
    ),
}


def caption_image_gemini(
    image_path: str,
    api_key: Optional[str] = None,
    model_alias: str = "Gemini-3.6-Flash",
    length_preset: str = "Medium",
    task_mode: str = "General",
    custom_system_prompt: Optional[str] = None,
) -> str:
    """Tạo caption cho 1 ảnh bằng Gemini API với task mode chuyên sâu."""
    key = get_api_key("gemini", api_key)
    if not key:
        raise ValueError("Chưa cấu hình GEMINI_API_KEY. Vui lòng nhập API Key hoặc cài đặt trong Colab Secrets.")

    from google import genai
    from google.genai import types

    model_name = GEMINI_MODELS.get(model_alias, model_alias)
    client = genai.Client(api_key=key)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    base_task_prompt = TASK_SYSTEM_PROMPTS.get(task_mode, TASK_SYSTEM_PROMPTS["General"])
    length_guide = LENGTH_PROMPTS.get(length_preset, LENGTH_PROMPTS["Medium"])

    rules = (
        "Do NOT include conversational preambles like 'Here is the caption' or 'The image depicts'. "
        "Always output clean English text only. "
    )

    if custom_system_prompt:
        prompt = f"{base_task_prompt} {custom_system_prompt}. {length_guide} {rules}"
    else:
        prompt = f"{base_task_prompt} {length_guide} {rules}"

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=types.Content(
                parts=[
                    types.Part(text=prompt),
                    types.Part(inline_data=types.Blob(data=image_bytes, mime_type="image/jpeg")),
                ]
            ),
        )
    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower() or "no longer available" in str(e).lower():
            # Tự động fallback sang default model mới nhất
            response = client.models.generate_content(
                model=DEFAULT_GEMINI_MODEL,
                contents=types.Content(
                    parts=[
                        types.Part(text=prompt),
                        types.Part(inline_data=types.Blob(data=image_bytes, mime_type="image/jpeg")),
                    ]
                ),
            )
        else:
            raise e

    return response.text.strip() if response.text else ""


def caption_video_gemini(
    video_path: str,
    api_key: Optional[str] = None,
    model_alias: str = "Gemini-3.6-Flash",
    length_preset: str = "Medium",
    custom_system_prompt: Optional[str] = None,
) -> str:
    """Tạo caption cho 1 video bằng Gemini API (truyền inline video bytes)."""
    key = get_api_key("gemini", api_key)
    if not key:
        raise ValueError("Chưa cấu hình GEMINI_API_KEY. Vui lòng nhập API Key hoặc cài đặt trong Colab Secrets.")

    from google import genai
    from google.genai import types

    model_name = GEMINI_MODELS.get(model_alias, model_alias)
    client = genai.Client(api_key=key)

    with open(video_path, "rb") as f:
        video_bytes = f.read()

    length_guide = LENGTH_PROMPTS.get(length_preset, LENGTH_PROMPTS["Medium"])
    base_prompt = (
        "You are an expert AI dataset captioner for Video Diffusion model training (Wan 2.1 / Wan 2.2). "
        "Describe the subject, primary actions, camera motion (pan, tilt, zoom, static), speed, lighting dynamics, and atmosphere. "
        "Do NOT include conversational preambles. Always output clean English text only. "
    )
    if custom_system_prompt:
        prompt = f"{base_prompt} {custom_system_prompt}. {length_guide}"
    else:
        prompt = f"{base_prompt} {length_guide}"

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=types.Content(
                parts=[
                    types.Part(text=prompt),
                    types.Part(inline_data=types.Blob(data=video_bytes, mime_type="video/mp4")),
                ]
            ),
        )
    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower() or "no longer available" in str(e).lower():
            response = client.models.generate_content(
                model=DEFAULT_GEMINI_MODEL,
                contents=types.Content(
                    parts=[
                        types.Part(text=prompt),
                        types.Part(inline_data=types.Blob(data=video_bytes, mime_type="video/mp4")),
                    ]
                ),
            )
        else:
            raise e

    return response.text.strip() if response.text else ""


def batch_caption_gemini(
    folder_path: str,
    api_key: Optional[str] = None,
    model_alias: str = "Gemini-3.6-Flash",
    length_preset: str = "Medium",
    task_mode: str = "General",
    custom_system_prompt: Optional[str] = None,
    overwrite: bool = False,
    is_video_folder: bool = False,
) -> int:
    """Chạy batch caption toàn bộ thư mục qua Gemini với task mode chuyên sâu."""
    model_name = GEMINI_MODELS.get(model_alias, model_alias)

    if is_video_folder:
        files = get_supported_videos(folder_path)
    else:
        files = get_supported_images(folder_path)

    if not files:
        print(f"[Thông báo] Không tìm thấy file trong thư mục: {folder_path}")
        return 0

    success_count = 0
    print(f"🚀 Bắt đầu sinh caption qua Gemini ({model_name}) | Chế độ: {task_mode} cho {len(files)} tệp...")

    for file_path in tqdm(files, desc=f"Gemini [{task_mode}]"):
        cap_path = os.path.splitext(file_path)[0] + ".txt"
        if os.path.exists(cap_path) and not overwrite and os.path.getsize(cap_path) > 0:
            continue

        try:
            if is_video_folder:
                caption = caption_video_gemini(
                    file_path,
                    api_key=api_key,
                    model_alias=model_alias,
                    length_preset=length_preset,
                    custom_system_prompt=custom_system_prompt,
                )
            else:
                caption = caption_image_gemini(
                    file_path,
                    api_key=api_key,
                    model_alias=model_alias,
                    length_preset=length_preset,
                    task_mode=task_mode,
                    custom_system_prompt=custom_system_prompt,
                )

            if caption:
                with open(cap_path, "w", encoding="utf-8") as f:
                    f.write(caption + "\n")
                success_count += 1

            time.sleep(0.5)

        except Exception as e:
            print(f"\n[Lỗi khi caption {file_path}]: {e}")
            time.sleep(2.0)

    print(f"✅ Hoàn thành caption: {success_count}/{len(files)} tệp đã được tạo mới.")
    return success_count
