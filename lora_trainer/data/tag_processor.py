"""
Tag & Caption Processor
Xử lý, định dạng, thêm/sửa/xóa trigger words, thẻ tag và caption tự động trong tập dữ liệu.
"""

import os
from pathlib import Path
from typing import Optional, Dict, List
from .cleaner import get_supported_images, get_supported_videos


def read_text_file(filepath: str) -> str:
    """Đọc nội dung tệp văn bản an toàn."""
    if not os.path.exists(filepath):
        return ""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="latin-1", errors="ignore") as f:
            return f.read().strip()


def write_text_file(filepath: str, content: str) -> None:
    """Ghi nội dung vào tệp văn bản."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")


def get_caption_path(media_path: str, ext: str = ".txt") -> str:
    """Trả về đường dẫn tệp caption tương ứng với media file."""
    base_name = os.path.splitext(media_path)[0]
    return f"{base_name}{ext}"


def process_tags(
    filename: str,
    custom_tag: str,
    append: bool = False,
    remove_tag: bool = False,
    replace_dict: Optional[Dict[str, str]] = None,
) -> None:
    """
    Xử lý tag trong một tệp caption:
    - remove_tag: Xóa tag custom_tag nếu có hoặc xóa sạch nếu custom_tag rỗng.
    - append: Thêm vào cuối caption (False = thêm vào đầu).
    - replace_dict: Thay thế các cụm từ theo bảng mapping.
    """
    current_content = read_text_file(filename)

    if remove_tag:
        if custom_tag:
            # Xóa custom_tag cụ thể
            tags = [t.strip() for t in current_content.split(",") if t.strip()]
            tags = [t for t in tags if t.lower() != custom_tag.lower()]
            new_content = ", ".join(tags)
        else:
            # Xóa toàn bộ
            new_content = ""
    else:
        if custom_tag:
            tags = [t.strip() for t in current_content.split(",") if t.strip()]
            
            # Tránh trùng lặp custom_tag nếu đã có
            if custom_tag.lower() not in [t.lower() for t in tags]:
                if append:
                    tags.append(custom_tag)
                else:
                    tags.insert(0, custom_tag)
            new_content = ", ".join(tags)
        else:
            new_content = current_content

    # Thực hiện search & replace nếu có
    if replace_dict and new_content:
        for old_str, new_str in replace_dict.items():
            new_content = new_content.replace(old_str, new_str)

    write_text_file(filename, new_content)


def process_dir_tags(
    image_dir: str,
    tag: str,
    append: bool = False,
    remove_tag: bool = False,
    replace_dict: Optional[Dict[str, str]] = None,
) -> int:
    """Áp dụng xử lý tag cho toàn bộ file caption trong thư mục ảnh/video."""
    if not os.path.exists(image_dir):
        return 0

    count = 0
    media_files = get_supported_images(image_dir) + get_supported_videos(image_dir)
    for media in media_files:
        cap_file = get_caption_path(media)
        process_tags(cap_file, tag, append=append, remove_tag=remove_tag, replace_dict=replace_dict)
        count += 1

    return count


def add_folder_name_tags(folder_path: str) -> int:
    """Lấy tên thư mục cha (sau khi làm sạch số repeats) và thêm vào làm tag đầu tiên."""
    if not os.path.exists(folder_path):
        return 0

    raw_folder_name = os.path.basename(os.path.normpath(folder_path))
    # Nếu thư mục có dạng 100_character_name -> lấy 'character_name'
    if "_" in raw_folder_name and raw_folder_name.split("_")[0].isdigit():
        folder_tag = "_".join(raw_folder_name.split("_")[1:])
    else:
        folder_tag = raw_folder_name

    folder_tag = folder_tag.replace("_", " ").strip()
    return process_dir_tags(folder_path, folder_tag, append=False, remove_tag=False)
