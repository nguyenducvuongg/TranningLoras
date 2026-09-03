"""
JoyCaption AI Vision Engine
Gán nhãn tự động chuẩn hóa sử dụng JoyCaption Alpha Two / Two local Transformers model.
Hỗ trợ tương thích hoàn hảo với transformers phiên bản mới (LlavaForConditionalGeneration).
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
        print(f"⚠️ Không tìm thấy ảnh hợp lệ tại: {folder_path}")
        return 0

    print(f"\n=======================================================")
    print(f"🤖 BẮT ĐẦU GÁN NHÃN JOYCAPTION")
    print(f"📂 Thư mục: {folder_path} ({len(images)} ảnh)")
    print(f"=======================================================\n")

    try:
        import torch
        from transformers import AutoProcessor
    except ImportError:
        print("⚠️ Cần cài đặt transformers và torch để chạy JoyCaption.")
        return 0

    model_id = "fancyfeast/llama-joycaption-alpha-two-hf-llava"
    model = None
    processor = None

    try:
        processor = AutoProcessor.from_pretrained(model_id)
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

        # Tối ưu hóa 4-bit qua BitsAndBytes cho GPU <= 20GB VRAM (Colab T4 / L4) để chạy cực mượt chỉ ~5.5GB
        quant_config = None
        if torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                vram_bytes = torch.cuda.get_device_properties(0).total_memory
                if vram_bytes < 22 * (1024 ** 3):
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    print("⚡ JoyCaption: Đã kích hoạt 4-bit NF4 Quantization tối ưu VRAM cho Colab!")
            except Exception:
                pass

        # Thử nạp với LlavaForConditionalGeneration (chuẩn cho LLaVA architecture)
        try:
            from transformers import LlavaForConditionalGeneration
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype if not quant_config else None,
                quantization_config=quant_config,
                device_map="auto" if torch.cuda.is_available() else None,
            )
        except Exception:
            from transformers import AutoModelForVision2Seq
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=dtype if not quant_config else None,
                quantization_config=quant_config,
                device_map="auto" if torch.cuda.is_available() else None,
            )

        if not torch.cuda.is_available():
            model = model.to("cpu")

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
            inputs = processor(text=prompt_text, images=[image], return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.5)

            # Cắt bỏ phần input prompt tokens để lấy phần trả lời mới
            input_len = inputs["input_ids"].shape[1]
            caption = processor.decode(output[0][input_len:], skip_special_tokens=True).strip()
            if caption:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)
                count += 1
        except Exception as e:
            print(f"⚠️ Lỗi JoyCaption tại {os.path.basename(img_path)}: {e}")

    print(f"🎉 Hoàn tất gán nhãn {count} ảnh với JoyCaption.")
    return count
