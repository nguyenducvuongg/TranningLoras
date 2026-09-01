"""
Prompt & Validation Sampler
Tự động trích xuất ngẫu nhiên caption hoặc trigger word từ dataset để làm validation sample prompt trong quá trình huấn luyện.
"""

import os
import random
from typing import Tuple, Optional


def get_random_sample_prompt(
    train_dir: str,
    control_dir: Optional[str] = None,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Lấy ngẫu nhiên một sample prompt từ các file caption .txt có sẵn trong train_dir.
    Trả về (prompt, sample_img_path, sample_control_img_path).
    """
    if not os.path.exists(train_dir):
        return "a high quality detailed photo of subject", None, None

    txt_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".txt")]
    if txt_files:
        chosen_txt = random.choice(txt_files)
        try:
            with open(chosen_txt, "r", encoding="utf-8") as f:
                prompt = f.read().strip()
                if prompt:
                    base = os.path.splitext(chosen_txt)[0]
                    # Tìm ảnh tương ứng
                    sample_img = None
                    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                        if os.path.exists(base + ext):
                            sample_img = base + ext
                            break

                    sample_ctrl = None
                    if control_dir and os.path.exists(control_dir) and sample_img:
                        ctrl_candidate = os.path.join(control_dir, os.path.basename(sample_img))
                        if os.path.exists(ctrl_candidate):
                            sample_ctrl = ctrl_candidate

                    return prompt, sample_img, sample_ctrl
        except Exception:
            pass

    # Fallback dựa trên tên thư mục
    folder_name = os.path.basename(os.path.normpath(train_dir))
    parts = folder_name.split("_", 1)
    tag = parts[1] if len(parts) == 2 and parts[0].isdigit() else folder_name

    return f"photo of {tag}, high quality, detailed, 8k", None, None
