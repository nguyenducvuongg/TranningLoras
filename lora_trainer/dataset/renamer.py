"""
Dataset Synchronizer & Renamer
Chuẩn hóa tên tệp đồng bộ cho ảnh, file caption (.txt) và thư mục paired control
theo thứ tự chuẩn hóa (ví dụ 0001.png, 0001.txt, 0001_control.png).
"""

import os
import shutil
from typing import List, Optional
from .cleaner import get_supported_images, IMAGE_EXTENSIONS


def standardize_single_folder(
    folder_path: str,
    prefix: str = "",
    digits: int = 4,
    auto_create_txt: bool = True,
    default_caption: str = "",
) -> int:
    """
    Chuẩn hóa tên toàn bộ ảnh và file .txt trong một thư mục:
    Đổi tên ảnh thành [prefix]0001.png, [prefix]0002.png... kèm file .txt tương ứng.
    Sử dụng cơ chế đổi tên trung gian an toàn để tránh ghi đè chéo.
    """
    if not os.path.exists(folder_path):
        return 0

    images = get_supported_images(folder_path)
    if not images:
        return 0

    # Bước 1: Thu thập cặp (ảnh, nội dung caption)
    items = []
    for img_path in images:
        base, ext = os.path.splitext(img_path)
        txt_path = base + ".txt"
        caption = default_caption
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    caption = f.read()
            except Exception:
                pass
        items.append({"orig_img": img_path, "ext": ext, "caption": caption})

    # Bước 2: Đổi tên trung gian tạm thời để chống conflict
    temp_pairs = []
    for idx, it in enumerate(items):
        tmp_img = os.path.join(folder_path, f"__tmp_renamer_{idx}{it['ext']}")
        shutil.move(it["orig_img"], tmp_img)
        # Xóa file txt cũ nếu có
        orig_txt = os.path.splitext(it["orig_img"])[0] + ".txt"
        if os.path.exists(orig_txt):
            try:
                os.remove(orig_txt)
            except Exception:
                pass
        temp_pairs.append({"tmp_img": tmp_img, "ext": it["ext"], "caption": it["caption"]})

    # Bước 3: Đổi tên chính thức theo định dạng chuẩn
    count = 0
    for idx, it in enumerate(temp_pairs, start=1):
        formatted_num = str(idx).zfill(digits)
        new_basename = f"{prefix}{formatted_num}"
        final_img = os.path.join(folder_path, f"{new_basename}{it['ext']}")
        shutil.move(it["tmp_img"], final_img)

        if auto_create_txt or it["caption"]:
            final_txt = os.path.join(folder_path, f"{new_basename}.txt")
            with open(final_txt, "w", encoding="utf-8") as f:
                f.write(it["caption"])

        count += 1

    print(f"🔢 Đã chuẩn hóa {count} tệp tại: {folder_path}")
    return count


def batch_standardize_datasets(
    train_folders: str,
    control_folders: Optional[str] = None,
    prefix: str = "",
    digits: int = 4,
    auto_create_txt: bool = True,
) -> None:
    """
    Chuẩn hóa đồng bộ cho danh sách các thư mục huấn luyện và thư mục Control đối chiếu 1-1.
    """
    t_dirs = [d.strip() for d in train_folders.split(",") if d.strip()]
    c_dirs = [d.strip() for d in control_folders.split(",") if d.strip()] if control_folders else []

    for t_dir in t_dirs:
        standardize_single_folder(t_dir, prefix=prefix, digits=digits, auto_create_txt=auto_create_txt)

    for c_dir in c_dirs:
        standardize_single_folder(c_dir, prefix=prefix, digits=digits, auto_create_txt=False)
