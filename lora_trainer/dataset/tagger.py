"""
Dataset Tag & Trigger Word Processor
Xử lý chèn từ khóa kích hoạt (Trigger word), gộp tag, loại bỏ tag trùng lặp
và tự động thêm tên thư mục làm tag cho toàn bộ file caption .txt.
"""

import os
from typing import List, Optional


def read_text_file(filepath: str) -> str:
    """Đọc an toàn file văn bản UTF-8."""
    if not os.path.exists(filepath):
        return ""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def write_text_file(filepath: str, content: str) -> None:
    """Ghi an toàn file văn bản UTF-8."""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        print(f"❌ Không thể ghi file {filepath}: {e}")


def process_tags(
    original_text: str,
    trigger_word: Optional[str] = None,
    position: str = "prefix", # 'prefix' hoặc 'suffix'
    remove_tags: Optional[List[str]] = None,
) -> str:
    """
    Xử lý danh sách tag / caption:
    - Chèn trigger word vào đầu (prefix) hoặc cuối (suffix)
    - Loại bỏ các tag không mong muốn
    - Khử trùng lặp tag
    """
    tags = [t.strip() for t in original_text.split(",") if t.strip()]

    if remove_tags:
        rem_set = {r.lower().strip() for r in remove_tags}
        tags = [t for t in tags if t.lower() not in rem_set]

    if trigger_word and trigger_word.strip():
        tw = trigger_word.strip()
        tags = [t for t in tags if t.lower() != tw.lower()]
        if position == "prefix":
            tags.insert(0, tw)
        else:
            tags.append(tw)

    # Loại bỏ trùng lặp giữ nguyên thứ tự
    seen = set()
    unique_tags = []
    for t in tags:
        t_low = t.lower()
        if t_low not in seen:
            seen.add(t_low)
            unique_tags.append(t)

    return ", ".join(unique_tags)


def process_dir_tags(
    dir_path: str,
    trigger_word: Optional[str] = None,
    position: str = "prefix",
    remove_tags: Optional[List[str]] = None,
) -> int:
    """Áp dụng xử lý tag cho toàn bộ file .txt trong một thư mục."""
    if not os.path.exists(dir_path):
        return 0

    count = 0
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.endswith(".txt"):
                txt_path = os.path.join(root, f)
                content = read_text_file(txt_path)
                updated = process_tags(content, trigger_word, position, remove_tags)
                write_text_file(txt_path, updated)
                count += 1

    print(f"🏷️ Đã cập nhật tag cho {count} file tại: {dir_path}")
    return count


def add_folder_name_tags(dir_path: str) -> None:
    """Lấy tên thư mục cha làm trigger word chèn vào đầu file caption."""
    folder_name = os.path.basename(os.path.normpath(dir_path))
    # Loại bỏ tiền tố số lặp ví dụ '20_character' -> 'character'
    parts = folder_name.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        tag = parts[1]
    else:
        tag = folder_name
    process_dir_tags(dir_path, trigger_word=tag, position="prefix")
