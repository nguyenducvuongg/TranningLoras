"""
JoyCaption Alpha Two AI Captioner
Mô hình JoyCaption chuyên biệt để tạo mô tả ảnh tự nhiên, phù hợp nhất cho FLUX và SDXL.
"""

import os
from typing import Optional, List, Tuple, Any
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


class ImageAdapter(_Module):
    def __init__(self, input_features: int, output_features: int, ln1: bool, pos_emb: bool, num_image_tokens: int, deep_extract: bool):
        super().__init__()
        self.deep_extract = deep_extract
        if self.deep_extract:
            input_features = input_features * 5
        self.linear1 = nn.Linear(input_features, output_features)
        self.act1 = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
        self.cls_token = None
        self.ln1 = nn.LayerNorm(input_features) if ln1 else nn.Identity()

    def forward(self, vision_outputs: Any):
        x = self.ln1(vision_outputs)
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        return x


_JOY_MODEL = None
_JOY_TOKENIZER = None
_JOY_CLIP = None
_JOY_PROCESSOR = None
_JOY_IMAGE_ADAPTER = None


def load_joycaption_pipeline(device: str = "cuda"):
    """Tải JoyCaption LLM và Clip Vision Adapter."""
    global _JOY_MODEL, _JOY_TOKENIZER, _JOY_CLIP, _JOY_PROCESSOR, _JOY_IMAGE_ADAPTER
    if _JOY_MODEL is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModel
        from huggingface_hub import hf_hub_download

        print("📦 Đang tải mô hình JoyCaption Alpha Two...")
        llm_id = "unsloth/Meta-Llama-3.1-8B-Instruct"
        clip_id = "google/siglip-so400m-patch14-384"
        adapter_repo = "fancyfeast/joy-caption-alpha-two"

        _JOY_TOKENIZER = AutoTokenizer.from_pretrained(llm_id, use_fast=True)
        _JOY_MODEL = AutoModelForCausalLM.from_pretrained(
            llm_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        _JOY_CLIP = AutoModel.from_pretrained(clip_id, torch_dtype=torch.bfloat16).to(device)
        _JOY_PROCESSOR = AutoProcessor.from_pretrained(clip_id)

        adapter_path = hf_hub_download(repo_id=adapter_repo, filename="image_adapter.pt")
        _JOY_IMAGE_ADAPTER = ImageAdapter(1152, 4096, True, False, 64, False)
        _JOY_IMAGE_ADAPTER.load_state_dict(torch.load(adapter_path, map_location=device))
        _JOY_IMAGE_ADAPTER.eval().to(device, dtype=torch.bfloat16)

    return _JOY_MODEL, _JOY_TOKENIZER, _JOY_CLIP, _JOY_PROCESSOR, _JOY_IMAGE_ADAPTER


def caption_image_joycaption(
    image_path: str,
    caption_type: str = "Descriptive",
    caption_length: str = "medium-length",
    device: str = "cuda",
) -> str:
    """Tạo caption cho 1 ảnh bằng JoyCaption."""
    model, tokenizer, clip, processor, adapter = load_joycaption_pipeline(device=device)

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        image_inputs = processor(images=img, return_tensors="pt").to(device)

    prompt = f"Write a {caption_length} {caption_type.lower()} caption for this image."
    convo = [
        {"role": "system", "content": "You are a helpful image captioner."},
        {"role": "user", "content": prompt},
    ]
    convo_str = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)

    with torch.inference_mode():
        vision_outputs = clip.vision_model(pixel_values=image_inputs.pixel_values)
        embedded_img = adapter(vision_outputs.last_hidden_state)

        inputs = tokenizer(convo_str, return_tensors="pt").to(device)
        input_embeds = model.get_input_embeddings()(inputs.input_ids)
        combined_embeds = torch.cat([embedded_img, input_embeds], dim=1)

        out = model.generate(
            inputs_embeds=combined_embeds,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            suppress_tokens=None,
        )

    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def batch_caption_joycaption(
    folder_path: str,
    caption_type: str = "Descriptive",
    caption_length: str = "medium-length",
    task_mode: str = "General",
    trigger_word: Optional[str] = None,
    overwrite: bool = False,
    device: str = "cuda",
    **kwargs,
) -> int:
    """Chạy batch caption toàn bộ thư mục bằng JoyCaption."""
    images = get_supported_images(folder_path)
    if not images:
        return 0

    load_joycaption_pipeline(device=device)
    success_count = 0
    print(f"🚀 Bắt đầu captioning bằng JoyCaption cho {len(images)} ảnh...")

    for img_path in tqdm(images, desc="JoyCaption"):
        cap_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(cap_path) and not overwrite and os.path.getsize(cap_path) > 0:
            continue

        try:
            caption = caption_image_joycaption(
                img_path, caption_type=caption_type, caption_length=caption_length, device=device
            )
            if caption:
                if trigger_word and trigger_word.strip() and trigger_word.lower() not in caption.lower():
                    caption = f"{trigger_word.strip()}, {caption}"
                with open(cap_path, "w", encoding="utf-8") as f:
                    f.write(caption + "\n")
                success_count += 1
        except Exception as e:
            print(f"\n[Lỗi JoyCaption {img_path}]: {e}")

    return success_count


# Alias tương thích
batch_caption_joy = batch_caption_joycaption
