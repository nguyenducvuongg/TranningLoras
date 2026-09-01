"""
Florence-2 Vision Captioning Engine
Gán nhãn tự động chuẩn hóa sử dụng mô hình Microsoft Florence-2 chạy cục bộ.
Khắc phục triệt để lỗi 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'
trên các phiên bản mới của thư viện transformers.
"""

import os
from typing import Optional
from PIL import Image
from tqdm import tqdm
from ..dataset.cleaner import get_supported_images


def batch_caption_florence(
    folder_path: str,
    task_mode: str = "General",
    trigger_word: Optional[str] = None,
    overwrite: bool = False,
) -> int:
    """Gán nhãn tự động với mô hình Microsoft Florence-2."""
    images = get_supported_images(folder_path)
    if not images:
        print(f"⚠️ Không tìm thấy ảnh hợp lệ tại: {folder_path}")
        return 0

    print(f"\n=======================================================")
    print(f"🤖 BẮT ĐẦU GÁN NHÃN FLORENCE-2")
    print(f"📂 Thư mục: {folder_path} ({len(images)} ảnh)")
    print(f"=======================================================")

    try:
        import torch
        from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM
    except ImportError:
        print("⚠️ Cần cài đặt transformers và torch để chạy Florence-2.")
        return 0

    model_id = "microsoft/Florence-2-large"
    model = None
    processor = None

    try:
        # Nạp và patch config để tương thích với transformers mới
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if hasattr(config, "text_config") and not hasattr(config.text_config, "forced_bos_token_id"):
            config.text_config.forced_bos_token_id = getattr(config.text_config, "bos_token_id", None)
        if not hasattr(config, "forced_bos_token_id"):
            config.forced_bos_token_id = getattr(config, "bos_token_id", None)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device)

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"❌ Không thể nạp Florence-2 ({e})")
        return 0

    task_prompt = "<MORE_DETAILED_CAPTION>" if task_mode != "Short" else "<CAPTION>"
    count = 0

    for img_path in tqdm(images, desc="🏷️ Florence-2"):
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
            inputs = processor(text=task_prompt, images=image, return_tensors="pt")
            
            processed_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    if v.dtype == torch.float32 and torch.cuda.is_available():
                        processed_inputs[k] = v.to(device, dtype=torch.float16)
                    else:
                        processed_inputs[k] = v.to(device)
                else:
                    processed_inputs[k] = v

            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=processed_inputs["input_ids"],
                    pixel_values=processed_inputs["pixel_values"],
                    max_new_tokens=256,
                    num_beams=3,
                )

            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(image.width, image.height),
            )

            caption = parsed_answer.get(task_prompt, "").strip()
            if trigger_word and trigger_word.strip():
                caption = f"{trigger_word.strip()}, {caption}"

            if caption:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)
                count += 1
        except Exception as e:
            print(f"⚠️ Lỗi Florence-2 tại {os.path.basename(img_path)}: {e}")

    print(f"🎉 Hoàn tất gán nhãn {count} ảnh với Florence-2.")
    return count
