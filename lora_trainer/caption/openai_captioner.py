"""
OpenAI Vision Captioner
Sinh caption ảnh chất lượng cao sử dụng OpenAI GPT Vision API (GPT-4o, GPT-5).
"""

import os
import io
import base64
import time
from typing import Optional, List
from PIL import Image
from tqdm import tqdm
from .key_manager import get_api_key
from ..data.cleaner import get_supported_images


def encode_image_base64(image_path: str) -> str:
    """Chuyển đổi file ảnh sang base64 chuỗi."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def caption_image_openai(
    image_path: str,
    api_key: Optional[str] = None,
    model_name: str = "gpt-4o",
    length_preset: str = "Medium",
    custom_system_prompt: Optional[str] = None,
) -> str:
    """Tạo caption cho 1 ảnh bằng OpenAI Vision."""
    key = get_api_key("openai", api_key)
    if not key:
        raise ValueError("Chưa cấu hình OPENAI_API_KEY. Vui lòng nhập API Key hoặc cài đặt trong Colab Secrets.")

    from openai import OpenAI
    client = OpenAI(api_key=key)

    base64_image = encode_image_base64(image_path)
    prompt = (
        "Describe this image concisely and objectively for image generation dataset training. "
        "Do NOT include conversational preambles. Output English only."
    )
    if custom_system_prompt:
        prompt = f"{prompt} {custom_system_prompt}"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()


def batch_caption_openai(
    folder_path: str,
    api_key: Optional[str] = None,
    model_name: str = "gpt-4o",
    length_preset: str = "Medium",
    custom_system_prompt: Optional[str] = None,
    overwrite: bool = False,
) -> int:
    """Chạy caption hàng loạt ảnh trong thư mục qua OpenAI."""
    images = get_supported_images(folder_path)
    if not images:
        return 0

    success_count = 0
    print(f"🚀 Bắt đầu sinh caption qua OpenAI ({model_name}) cho {len(images)} ảnh...")

    for img_path in tqdm(images, desc="OpenAI Captioning"):
        cap_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(cap_path) and not overwrite and os.path.getsize(cap_path) > 0:
            continue

        try:
            caption = caption_image_openai(
                img_path, api_key, model_name, length_preset, custom_system_prompt
            )
            if caption:
                with open(cap_path, "w", encoding="utf-8") as f:
                    f.write(caption + "\n")
                success_count += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"\n[Lỗi OpenAI {img_path}]: {e}")
            time.sleep(2.0)

    return success_count
