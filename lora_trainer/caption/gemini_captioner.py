"""
Gemini Multimodal AI Captioner
Tận dụng sức mạnh của Google Gemini 2.5 (Pro, Flash, Flash-Lite) để sinh caption chất lượng cao cho cả Hình ảnh và Video.
"""

import os
import io
import time
from typing import Optional, List
from PIL import Image
from tqdm import tqdm
from .key_manager import get_api_key
from ..data.cleaner import get_supported_images, get_supported_videos

GEMINI_MODELS = {
    "Gemini-2.5-Flash": "gemini-2.5-flash",
    "Gemini-2.5-Pro": "gemini-2.5-pro",
    "Gemini-2.5-Flash-Lite": "gemini-2.5-flash-lite",
}

LENGTH_PROMPTS = {
    "Short": "Write a concise single-sentence caption describing the main subject and action, around 20-30 words.",
    "Medium": "Write a detailed caption describing the subject, style, lighting, composition, and background, around 50-70 words.",
    "Long": "Write a comprehensive and extremely detailed descriptive caption explaining every element, lighting, texture, and action, around 100-150 words.",
}


def caption_image_gemini(
    image_path: str,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash",
    length_preset: str = "Medium",
    custom_system_prompt: Optional[str] = None,
) -> str:
    """Tạo caption cho 1 ảnh bằng Gemini API."""
    key = get_api_key("gemini", api_key)
    if not key:
        raise ValueError("Chưa cấu hình GEMINI_API_KEY. Vui lòng nhập API Key hoặc cài đặt trong Colab Secrets.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)
    
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    length_guide = LENGTH_PROMPTS.get(length_preset, LENGTH_PROMPTS["Medium"])
    base_prompt = (
        "You are an expert AI dataset captioner for Diffusion model training. "
        "Describe the visual content objectively and clearly. "
        "Do NOT include conversational preambles like 'Here is the caption' or 'The image shows'. "
        "Always output clean English text only. "
    )
    if custom_system_prompt:
        prompt = f"{base_prompt} {custom_system_prompt}. {length_guide}"
    else:
        prompt = f"{base_prompt} {length_guide}"

    response = client.models.generate_content(
        model=model_name,
        contents=types.Content(
            parts=[
                types.Part(text=prompt),
                types.Part(inline_data=types.Blob(data=image_bytes, mime_type="image/jpeg")),
            ]
        ),
    )
    return response.text.strip() if response.text else ""


def caption_video_gemini(
    video_path: str,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash",
    length_preset: str = "Medium",
    custom_system_prompt: Optional[str] = None,
) -> str:
    """Tạo caption cho 1 video bằng Gemini API (truyền inline video bytes)."""
    key = get_api_key("gemini", api_key)
    if not key:
        raise ValueError("Chưa cấu hình GEMINI_API_KEY. Vui lòng nhập API Key hoặc cài đặt trong Colab Secrets.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)

    with open(video_path, "rb") as f:
        video_bytes = f.read()

    length_guide = LENGTH_PROMPTS.get(length_preset, LENGTH_PROMPTS["Medium"])
    base_prompt = (
        "You are an expert AI dataset captioner for Video Diffusion model training (like Wan 2.1). "
        "Describe the main actions, camera movements, subject transformations, and atmosphere throughout the video. "
        "Do NOT include conversational preambles. "
        "Always output clean English text only. "
    )
    if custom_system_prompt:
        prompt = f"{base_prompt} {custom_system_prompt}. {length_guide}"
    else:
        prompt = f"{base_prompt} {length_guide}"

    response = client.models.generate_content(
        model=model_name,
        contents=types.Content(
            parts=[
                types.Part(text=prompt),
                types.Part(inline_data=types.Blob(data=video_bytes, mime_type="video/mp4")),
            ]
        ),
    )
    return response.text.strip() if response.text else ""


def batch_caption_gemini(
    folder_path: str,
    api_key: Optional[str] = None,
    model_alias: str = "Gemini-2.5-Flash",
    length_preset: str = "Medium",
    custom_system_prompt: Optional[str] = None,
    overwrite: bool = False,
    is_video_folder: bool = False,
) -> int:
    """Chạy caption hàng loạt cho toàn bộ file trong thư mục qua Gemini."""
    model_name = GEMINI_MODELS.get(model_alias, model_alias)
    
    if is_video_folder:
        files = get_supported_videos(folder_path)
    else:
        files = get_supported_images(folder_path)

    if not files:
        print(f"[Thông báo] Không tìm thấy file trong thư mục: {folder_path}")
        return 0

    success_count = 0
    print(f"🚀 Bắt đầu sinh caption qua Gemini ({model_name}) cho {len(files)} tệp...")

    for file_path in tqdm(files, desc="Gemini Captioning"):
        cap_path = os.path.splitext(file_path)[0] + ".txt"
        if os.path.exists(cap_path) and not overwrite and os.path.getsize(cap_path) > 0:
            continue

        try:
            if is_video_folder:
                caption = caption_video_gemini(
                    file_path, api_key, model_name, length_preset, custom_system_prompt
                )
            else:
                caption = caption_image_gemini(
                    file_path, api_key, model_name, length_preset, custom_system_prompt
                )

            if caption:
                with open(cap_path, "w", encoding="utf-8") as f:
                    f.write(caption + "\n")
                success_count += 1

            # Sleep nhẹ để tránh chạm quota limit
            time.sleep(0.5)

        except Exception as e:
            print(f"\n[Lỗi khi caption {file_path}]: {e}")
            time.sleep(2.0)

    print(f"✅ Hoàn thành caption: {success_count}/{len(files)} tệp đã được tạo mới.")
    return success_count
