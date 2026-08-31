"""
Sample & Resolution Utilities
Lấy mẫu ngẫu nhiên prompt/hình ảnh phục vụ xem trước (Preview validation) và tính toán kích thước ảnh.
"""

import os
import random
from typing import Tuple, Optional
from PIL import Image
from ..data.cleaner import get_supported_images


def calculate_bucket_resolution(
    image_path: str, max_size: int = 1536, divisible_by: int = 64
) -> Tuple[int, int]:
    """
    Tính toán độ phân giải tối ưu cho ảnh, đảm bảo tỉ lệ và chia hết cho divisible_by.
    """
    if not os.path.exists(image_path):
        return (1024, 1024)

    with Image.open(image_path) as img:
        width, height = img.size

    # Tính toán scale nếu vượt quá max_size
    scale = min(max_size / max(width, height), 1.0)
    target_w = int(width * scale)
    target_h = int(height * scale)

    # Làm tròn về bội số của divisible_by
    target_w = (target_w // divisible_by) * divisible_by
    target_h = (target_h // divisible_by) * divisible_by

    return (max(divisible_by, target_w), max(divisible_by, target_h))


def get_random_sample_prompt(
    folder_path: str, control_folder_path: Optional[str] = None
) -> Tuple[str, str, Optional[str]]:
    """
    Lấy ngẫu nhiên một mẫu dữ liệu (prompt caption, image path, control image path nếu có).
    Trả về (prompt, image_path, control_image_path).
    """
    images = get_supported_images(folder_path)
    if not images:
        return ("a high quality photo", "", None)

    sampled_img = random.choice(images)
    cap_file = os.path.splitext(sampled_img)[0] + ".txt"

    prompt = "a high quality photo"
    if os.path.exists(cap_file):
        try:
            with open(cap_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    prompt = content
        except Exception:
            pass

    ctrl_img = None
    if control_folder_path and os.path.exists(control_folder_path):
        rel_name = os.path.basename(sampled_img)
        expected_ctrl = os.path.join(control_folder_path, rel_name)
        if os.path.exists(expected_ctrl):
            ctrl_img = expected_ctrl

    return (prompt, sampled_img, ctrl_img)
