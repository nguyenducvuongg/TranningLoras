"""
OpenAI Vision Captioning Engine
Gán nhãn tự động chuẩn hóa sử dụng GPT-4o / GPT-4o-mini Vision API.
"""

import os
import base64
import time
from typing import Optional
from tqdm import tqdm
from ..core.key_vault import get_api_key
from ..dataset.cleaner import get_supported_images
from .base_captioner import build_task_prompt


def encode_image(image_path: str) -> str:
    """Mã hóa ảnh thành Base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def batch_caption_openai(
    folder_path: str,
    api_key: Optional[str] = None,
    model_name: str = "gpt-4o",
    task_mode: str = "General",
    caption_length: str = "Medium",
    trigger_word: Optional[str] = None,
    overwrite: bool = False,
) -> int:
    """Gán nhãn tự động với OpenAI GPT-4o Vision API."""
    effective_key = api_key or get_api_key("openai")
    if not effective_key:
        print("❌ Lỗi: Chưa cấu hình OpenAI API Key trong Vault!")
        return 0

    images = get_supported_images(folder_path)
    if not images:
        return 0

    try:
        from openai import OpenAI
        client = OpenAI(api_key=effective_key)
    except ImportError:
        print("📦 Cài đặt thư viện openai...")
        import subprocess
        subprocess.run(["pip", "install", "-q", "openai"], check=False)
        from openai import OpenAI
        client = OpenAI(api_key=effective_key)

    prompt = build_task_prompt(task_mode, caption_length, trigger_word)
    count = 0

    print(f"\n=======================================================")
    print(f"🤖 BẮT ĐẦU GÁN NHÃN OPENAI ({model_name})")
    print(f"📂 Thư mục: {folder_path} ({len(images)} ảnh)")
    print(f"=======================================================\n")

    for img_path in tqdm(images, desc=f"🏷️ OpenAI ({model_name})"):
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(txt_path) and not overwrite:
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    if f.read().strip():
                        continue
            except Exception:
                pass

        try:
            b64_img = encode_image(img_path)
            ext = os.path.splitext(img_path)[1].lower().replace(".", "")
            mime = "jpeg" if ext in ["jpg", "jpeg"] else ext

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64_img}"}},
                        ],
                    }
                ],
                max_tokens=300,
            )

            caption = response.choices[0].message.content.strip().replace("\n", " ")
            if caption:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)
                count += 1
                time.sleep(0.3)
        except Exception as e:
            print(f"⚠️ Lỗi OpenAI tại {os.path.basename(img_path)}: {e}")

    print(f"🎉 Hoàn tất gán nhãn {count} ảnh với OpenAI.")
    return count
