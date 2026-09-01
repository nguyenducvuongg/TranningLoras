"""
JoyCaption Alpha Two AI Captioner
Mô hình JoyCaption chuyên biệt để tạo mô tả ảnh tự nhiên, phù hợp nhất cho FLUX, SDXL và MidJourney style.
Tự động nạp định dạng LLaVA chuẩn hoặc nạp adapter fallback từ các kho mirror công khai,
tối ưu hóa bộ nhớ VRAM với 4-bit Quantization (chạy mượt mà trên cả GPU T4, L4, A100).
"""

import os
import gc
from typing import Optional, List, Tuple, Any, Dict
from PIL import Image
from tqdm import tqdm
from ..data.cleaner import get_supported_images

try:
    import torch
    import torch.nn as nn
    _Module = nn.Module
except ImportError:
    torch = None
    _Module = object


# ==============================================================================
# 1. CẤU HÌNH PROMPT THEO DẠNG CAPTION & MỤC ĐÍCH HUẤN LUYỆN
# ==============================================================================

JOY_LENGTH_WORDS = {
    "short": 30,
    "medium": 70,
    "long": 130,
}

JOY_TASK_PROMPTS = {
    "General": "Write a {length} descriptive caption for this image in a natural tone within {words} words.",
    "Skin_Portrait": "Write a {length} detailed description focusing on the person, facial features, skin texture, expression, lighting, and pose within {words} words.",
    "Upscale_Restoration": "Describe the subject, fine textures, lighting, color details, and camera focus in high clarity within {words} words.",
    "Art_Style": "Analyze the artistic style, color palette, brushwork, composition, mood, and aesthetic of this image within {words} words.",
    "Character_Outfit": "Describe the character, hairstyle, clothes, accessories, materials, and overall outfit in detail within {words} words.",
}


# ==============================================================================
# 2. KHỞI TẠO & NẠP MODEL TỐI ƯU HÓA VRAM (4-BIT / BFLOAT16)
# ==============================================================================

_JOY_PIPELINE = None


def load_joycaption_model(
    model_id: str = "fancyfeast/llama-joycaption-alpha-two-hf-llava",
    device: str = "cuda",
    load_in_4bit: bool = True,
):
    """
    Tải JoyCaption Alpha Two với cơ chế 4-bit Quantization chống tràn VRAM.
    Tự động thử các mirror công khai nếu gặp sự cố kết nối.
    """
    global _JOY_PIPELINE
    if _JOY_PIPELINE is not None:
        return _JOY_PIPELINE

    if torch is None or not torch.cuda.is_available():
        device = "cpu"

    from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

    # Danh sách các mirror công khai chuẩn
    candidate_models = [
        model_id,
        "fancyfeast/llama-joycaption-alpha-two-hf-llava",
        "BullseyeMxP/joy-caption-alpha-two",
        "John6666/joy-caption-alpha-two-cli-mod",
        "camenduru/joy-caption-alpha-two",
    ]

    last_error = None
    for cand in candidate_models:
        try:
            print(f"📦 Đang nạp mô hình JoyCaption ({cand})...")
            processor = AutoProcessor.from_pretrained(cand, trust_remote_code=True)
            
            if device == "cuda" and load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                model = LlavaForConditionalGeneration.from_pretrained(
                    cand,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                    trust_remote_code=True,
                )
            else:
                dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device == "cuda" else torch.float32)
                model = LlavaForConditionalGeneration.from_pretrained(
                    cand,
                    torch_dtype=dtype,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True,
                )
                if device != "cuda":
                    model = model.to(device)

            model.eval()
            _JOY_PIPELINE = {"model": model, "processor": processor, "device": device}
            print("✅ Đã sẵn sàng JoyCaption Alpha Two!")
            return _JOY_PIPELINE
        except Exception as e:
            last_error = e
            print(f"⚠️ Không thể tải từ {cand}: {e}. Đang thử mirror tiếp theo...")
            continue

    raise RuntimeError(f"Không thể khởi tạo JoyCaption từ bất kỳ nguồn mirror nào: {last_error}")


# ==============================================================================
# 3. HÀM CAPTION CHO 1 ẢNH
# ==============================================================================

def caption_image_joycaption(
    image_path: str,
    caption_length: str = "Medium",
    task_mode: str = "General",
    device: str = "cuda",
) -> str:
    """Tạo caption chuyên sâu cho 1 ảnh bằng JoyCaption."""
    pipe = load_joycaption_model(device=device)
    model = pipe["model"]
    processor = pipe["processor"]
    dev = pipe["device"]

    norm_len = caption_length.lower().strip()
    if "short" in norm_len:
        len_key = "short"
    elif "long" in norm_len:
        len_key = "long"
    else:
        len_key = "medium"

    words = JOY_LENGTH_WORDS.get(len_key, 70)
    template = JOY_TASK_PROMPTS.get(task_mode, JOY_TASK_PROMPTS["General"])
    user_prompt = template.format(length=len_key, words=words)

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        
        # Áp dụng chat template chuẩn của Llava / JoyCaption
        messages = [
            {"role": "user", "content": f"<image>\n{user_prompt}"}
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=img, return_tensors="pt").to(dev)

    max_tokens = 60 if len_key == "short" else (140 if len_key == "medium" else 250)

    with torch.inference_mode():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    # Cắt bỏ phần prompt đầu vào, chỉ lấy output sinh ra
    input_len = inputs["input_ids"].shape[1]
    output_tokens = generate_ids[0][input_len:]
    caption = processor.decode(output_tokens, skip_special_tokens=True).strip()

    # Dọn dẹp các ký tự thừa
    caption = caption.replace("<|eot_id|>", "").replace("<|start_header_id|>", "").strip()
    return caption


# ==============================================================================
# 4. HÀM CHẠY HÀNG LOẠT (BATCH PROCESSING)
# ==============================================================================

def batch_caption_joycaption(
    folder_path: str,
    caption_length: str = "Medium",
    task_mode: str = "General",
    trigger_word: Optional[str] = None,
    overwrite: bool = False,
    device: str = "cuda",
    **kwargs,
) -> int:
    """Chạy batch caption toàn bộ thư mục bằng JoyCaption Alpha Two."""
    images = get_supported_images(folder_path)
    if not images:
        return 0

    load_joycaption_model(device=device)
    success_count = 0
    print(f"🚀 Bắt đầu captioning bằng JoyCaption ({task_mode} - {caption_length}) cho {len(images)} ảnh...")

    for i, img_path in enumerate(tqdm(images, desc="JoyCaption")):
        cap_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(cap_path) and not overwrite and os.path.getsize(cap_path) > 0:
            continue

        try:
            caption = caption_image_joycaption(
                img_path, caption_length=caption_length, task_mode=task_mode, device=device
            )
            if caption:
                if trigger_word and trigger_word.strip() and trigger_word.lower() not in caption.lower():
                    caption = f"{trigger_word.strip()}, {caption}"
                with open(cap_path, "w", encoding="utf-8") as f:
                    f.write(caption + "\n")
                success_count += 1
        except Exception as e:
            print(f"\n[Lỗi JoyCaption {os.path.basename(img_path)}]: {e}")

        # Định kỳ dọn dẹp GPU cache
        if (i + 1) % 25 == 0 and torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    return success_count


# Alias tương thích
batch_caption_joy = batch_caption_joycaption
