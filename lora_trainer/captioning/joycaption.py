"""
JoyCaption AI Vision Engine
Gán nhãn tự động chuẩn hóa sử dụng JoyCaption Alpha Two / Two local Transformers model.
"""

import os
from typing import Optional
from PIL import Image
from tqdm import tqdm
from ..dataset.cleaner import get_supported_images
from .base_captioner import build_task_prompt


def batch_caption_joy(
    folder_path: str,
    task_mode: str = "General",
    caption_length: str = "Medium",
    trigger_word: Optional[str] = None,
    overwrite: bool = False,
) -> int:
    """Gán nhãn tự động với mô hình JoyCaption chạy cục bộ trên GPU."""
    images = get_supported_images(folder_path)
    if not images:
        return 0

    print(f"\n=======================================================")
    print(f"🤖 BẮT ĐẦU GÁN NHÃN JOYCAPTION")
    print(f"📂 Thư mục: {folder_path} ({len(images)} ảnh)")
    print(f"=======================================================\n")

    try:
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
    except ImportError:
        print("⚠️ Cần cài đặt transformers và torch để chạy JoyCaption.")
        return 0

    model_id = "fancyfeast/llama-joycaption-alpha-two-hf-llava"
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
        )
    except Exception as e:
        print(f"❌ Không thể nạp JoyCaption ({e})")
        return 0

    prompt = build_task_prompt(task_mode, caption_length, trigger_word)
    count = 0

    for img_path in tqdm(images, desc="🏷️ JoyCaption"):
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(txt_path) and not overwrite:
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    if f.read().strip():
                        continue
            except Exception:
                pass

        try:
            image = Image.open(img_path).convert("RGB")
            convo = [{"role": "user", "content": f"{prompt}\n<image>"}]
            prompt_text = processor.apply_chat_template(convo, add_generation_prompt=True)
            inputs = processor(text=prompt_text, images=[image], return_tensors="pt").to(model.device)

            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.5)
            
            caption = processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            if caption:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)
                count += 1
        except Exception as e:
            print(f"⚠️ Lỗi JoyCaption tại {os.path.basename(img_path)}: {e}")

    print(f"🎉 Hoàn tất gán nhãn {count} ảnh với JoyCaption.")
    return count
