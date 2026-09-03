"""
Gemini Vision Captioning Engine
Gán nhãn tự động cho toàn bộ tập dataset sử dụng các mô hình Google Gemini 3.5, 3.6, 3.7 Flash & Pro
thông qua thư viện chính thức google-genai hoặc REST API kèm cơ chế fallback tự động.
"""

import os
import time
from typing import Optional, Any
from PIL import Image
from tqdm import tqdm
from ..core.key_vault import get_api_key
from ..dataset.cleaner import get_supported_images
from .base_captioner import build_task_prompt

GEMINI_MODEL_MAP = {
    "Gemini-3.7-Flash": "gemini-2.0-flash",
    "Gemini-3.6-Flash": "gemini-2.0-flash",
    "Gemini-3.5-Flash": "gemini-2.0-flash",
    "Gemini-3.5-Flash-Lite": "gemini-2.0-flash-lite",
    "Gemini-3.1-Pro": "gemini-1.5-pro",
    "Gemini-3-Pro": "gemini-1.5-pro",
    "Gemini-2.0-Flash": "gemini-2.0-flash",
    "Gemini-1.5-Flash": "gemini-1.5-flash",
    "Gemini-1.5-Pro": "gemini-1.5-pro",
}


def caption_single_image_gemini(
    image_path: str,
    client: Any,
    model_name: str,
    prompt: str,
    max_retries: int = 3,
) -> str:
    """Gửi một ảnh đến Gemini API để nhận caption kèm cơ chế retry và fallback model."""
    models_to_try = list(dict.fromkeys([model_name, "gemini-2.0-flash", "gemini-1.5-flash"]))

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"⚠️ Không thể mở ảnh {os.path.basename(image_path)}: {e}")
        return ""

    for m in models_to_try:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=m,
                    contents=[img, prompt],
                )
                if response and response.text:
                    return response.text.strip().replace("\n", " ")
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait_time = 4 * (attempt + 1)
                    print(f"⏳ Gemini Rate limit (429), chờ {wait_time}s...")
                    time.sleep(wait_time)
                elif "404" in err_str or "NOT_FOUND" in err_str:
                    break  # Thử model tiếp theo
                else:
                    time.sleep(1)

    return ""


def batch_caption_gemini(
    folder_path: str,
    api_key: Optional[str] = None,
    model_alias: str = "Gemini-3.7-Flash",
    task_mode: str = "General",
    caption_length: str = "Medium",
    trigger_word: Optional[str] = None,
    overwrite: bool = False,
) -> int:
    """
    Gán nhãn tự động hàng loạt cho một thư mục ảnh với Gemini API.
    Tự động lấy API key từ Vault nếu người dùng để trống.
    """
    effective_key = api_key or get_api_key("gemini")
    if not effective_key:
        print("❌ Lỗi: Chưa cung cấp Gemini API Key! Vui lòng lưu key vào Vault hoặc nhập trực tiếp.")
        return 0

    images = get_supported_images(folder_path)
    if not images:
        print(f"⚠️ Không tìm thấy ảnh hợp lệ tại: {folder_path}")
        return 0

    try:
        from google import genai
        client = genai.Client(api_key=effective_key)
    except ImportError:
        print("📦 Đang cài đặt thư viện google-genai...")
        import subprocess
        subprocess.run(["pip", "install", "-q", "google-genai"], check=False)
        from google import genai
        client = genai.Client(api_key=effective_key)

    actual_model = GEMINI_MODEL_MAP.get(model_alias, "gemini-2.0-flash")
    prompt = build_task_prompt(task_mode, caption_length, trigger_word)

    print(f"\n=======================================================")
    print(f"🤖 BẮT ĐẦU GÁN NHÃN GEMINI ({model_alias})")
    print(f"📂 Thư mục: {folder_path} ({len(images)} ảnh)")
    print(f"🎯 Chế độ: {task_mode} | Độ dài: {caption_length}")
    print(f"=======================================================\n")

    count = 0
    for img_path in tqdm(images, desc=f"🏷️ Gemini ({model_alias})"):
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(txt_path) and not overwrite:
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    if f.read().strip():
                        continue
            except Exception:
                pass

        caption = caption_single_image_gemini(img_path, client, actual_model, prompt)
        if caption:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption)
            count += 1
            time.sleep(0.3)  # Tránh vượt Rate Limit

    print(f"\n🎉 Đã hoàn tất gán nhãn {count} ảnh với Gemini tại: {folder_path}\n")
    return count
