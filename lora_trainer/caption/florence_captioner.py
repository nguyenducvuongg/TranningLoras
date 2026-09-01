"""
Florence-2 Local AI Captioner
Mô hình Vision-Language của Microsoft chạy offline trực tiếp trên GPU Colab, không cần API Key,
tối ưu hóa tốc độ cao với SDPA attention và cơ chế fallback tự động.
"""

import os
import gc
from typing import Optional, List, Any
from PIL import Image
from tqdm import tqdm
from ..data.cleaner import get_supported_images

try:
    import torch
except ImportError:
    torch = None

_FLORENCE_MODEL = None
_FLORENCE_PROCESSOR = None

FLORENCE_TASK_PROMPTS = {
    "Short": "<CAPTION>",
    "Medium": "<DETAILED_CAPTION>",
    "Long": "<MORE_DETAILED_CAPTION>",
}


def load_florence_model(
    model_id: str = "microsoft/Florence-2-large",
    device: str = "cuda",
):
    """Khởi tạo và tải model Florence-2 vào bộ nhớ GPU với cơ chế fallback tự động."""
    global _FLORENCE_MODEL, _FLORENCE_PROCESSOR
    if _FLORENCE_MODEL is not None and _FLORENCE_PROCESSOR is not None:
        return _FLORENCE_MODEL, _FLORENCE_PROCESSOR

    if torch is None or not torch.cuda.is_available():
        device = "cpu"

    from transformers import AutoProcessor, AutoModelForCausalLM

    candidates = [
        model_id,
        "microsoft/Florence-2-large",
        "microsoft/Florence-2-base",
        "thwri/Florence-2-large-FT-DocVQA",
    ]

    last_error = None
    dtype = torch.float16 if device == "cuda" else torch.float32

    for cand in candidates:
        try:
            print(f"📦 Đang tải Florence-2 Model ({cand})...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    cand,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    attn_implementation="sdpa",
                ).to(device)
            except Exception:
                model = AutoModelForCausalLM.from_pretrained(
                    cand,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                ).to(device)

            processor = AutoProcessor.from_pretrained(
                cand,
                trust_remote_code=True,
            )

            model.eval()
            _FLORENCE_MODEL = model
            _FLORENCE_PROCESSOR = processor
            print("✅ Đã sẵn sàng Florence-2!")
            return _FLORENCE_MODEL, _FLORENCE_PROCESSOR
        except Exception as e:
            last_error = e
            print(f"⚠️ Không thể tải {cand}: {e}. Đang thử model fallback...")
            continue

    raise RuntimeError(f"Không thể khởi tạo Florence-2 từ bất kỳ model nào: {last_error}")


def caption_image_florence(
    image_path: str,
    task_preset: str = "Medium",
    device: str = "cuda",
) -> str:
    """Tạo caption cho 1 ảnh bằng Florence-2."""
    model, processor = load_florence_model(device=device)

    # Chuẩn hóa prompt
    preset_key = "Medium"
    norm_preset = task_preset.capitalize()
    if norm_preset in FLORENCE_TASK_PROMPTS:
        preset_key = norm_preset
    elif "short" in task_preset.lower():
        preset_key = "Short"
    elif "long" in task_preset.lower():
        preset_key = "Long"

    prompt = FLORENCE_TASK_PROMPTS.get(preset_key, "<DETAILED_CAPTION>")

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        dtype = torch.float16 if device == "cuda" else torch.float32
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(device, dtype)

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, task=prompt, image_size=(img.width, img.height)
    )

    caption = parsed_answer.get(prompt, "")
    return str(caption).strip()


def batch_caption_florence(
    folder_path: str,
    task_preset: str = "Medium",
    caption_length: Optional[str] = None,
    task_mode: str = "General",
    trigger_word: Optional[str] = None,
    overwrite: bool = False,
    device: str = "cuda",
    **kwargs,
) -> int:
    """Chạy batch caption toàn bộ thư mục bằng Florence-2."""
    effective_length = caption_length or task_preset or "Medium"
    images = get_supported_images(folder_path)
    if not images:
        return 0

    load_florence_model(device=device)
    success_count = 0
    print(f"🚀 Bắt đầu captioning bằng Florence-2 ({effective_length}) cho {len(images)} ảnh...")

    for i, img_path in enumerate(tqdm(images, desc="Florence-2")):
        cap_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(cap_path) and not overwrite and os.path.getsize(cap_path) > 0:
            continue

        try:
            caption = caption_image_florence(img_path, task_preset=effective_length, device=device)
            if caption:
                if trigger_word and trigger_word.strip() and trigger_word.lower() not in caption.lower():
                    caption = f"{trigger_word.strip()}, {caption}"
                with open(cap_path, "w", encoding="utf-8") as f:
                    f.write(caption + "\n")
                success_count += 1
        except Exception as e:
            print(f"\n[Lỗi Florence {os.path.basename(img_path)}]: {e}")

        # Định kỳ dọn dẹp GPU cache
        if (i + 1) % 30 == 0 and torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    return success_count
