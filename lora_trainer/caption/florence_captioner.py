"""
Florence-2 Local AI Captioner
Mô hình Vision chạy offline miễn phí trực tiếp trên GPU Colab không cần API key.
"""

import os
from typing import Optional, List
from PIL import Image
from tqdm import tqdm
from ..data.cleaner import get_supported_images

_FLORENCE_MODEL = None
_FLORENCE_PROCESSOR = None

TASK_PROMPTS = {
    "Short": "<CAPTION>",
    "Medium": "<DETAILED_CAPTION>",
    "Long": "<MORE_DETAILED_CAPTION>",
}


def load_florence_model(model_id: str = "microsoft/Florence-2-large", device: str = "cuda"):
    """Khởi tạo và tải model Florence-2 vào bộ nhớ GPU."""
    global _FLORENCE_MODEL, _FLORENCE_PROCESSOR
    if _FLORENCE_MODEL is None:
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        print(f"📦 Đang tải Florence-2 Model ({model_id})...")
        dtype = torch.float16 if device == "cuda" else torch.float32
        _FLORENCE_MODEL = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)
        _FLORENCE_PROCESSOR = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
    return _FLORENCE_MODEL, _FLORENCE_PROCESSOR


def caption_image_florence(
    image_path: str,
    task_preset: str = "Medium",
    device: str = "cuda",
) -> str:
    """Tạo caption cho 1 ảnh bằng Florence-2."""
    model, processor = load_florence_model(device=device)
    prompt = TASK_PROMPTS.get(task_preset, "<DETAILED_CAPTION>")

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(device, torch.float16 if device == "cuda" else torch.float32)

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

    for img_path in tqdm(images, desc="Florence-2 Captioning"):
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
            print(f"\n[Lỗi Florence {img_path}]: {e}")

    return success_count
