"""
Data Cleaner & Validator
Dọn dẹp các tệp rác hệ thống (macOS, Windows, Linux) và kiểm tra tính toàn vẹn của tệp ảnh/video.
"""

import os
import shutil
from pathlib import Path
from typing import List, Set, Tuple
from PIL import Image

IMAGE_EXTENSIONS: Set[str] = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif",
    ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP", ".TIFF", ".TIF"
}

VIDEO_EXTENSIONS: Set[str] = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv",
    ".MP4", ".MOV", ".AVI", ".MKV", ".WEBM", ".FLV"
}

JUNK_FILENAMES: Set[str] = {
    ".DS_Store", "Thumbs.db", "desktop.ini", ".directory"
}


def clean_directory(directory_path: str) -> Tuple[int, int]:
    """
    Dọn dẹp thư mục: xóa file rác hệ thống, file ẩn `._*`, và file 0-byte.
    Trả về tuple (số file rác đã xóa, số file hợp lệ còn lại).
    """
    if not os.path.exists(directory_path):
        return (0, 0)

    removed_count = 0
    valid_count = 0

    for root, dirs, files in os.walk(directory_path, topdown=False):
        for f in files:
            full_path = os.path.join(root, f)
            
            # Xóa file rác hệ điều hành
            is_junk = (
                f in JUNK_FILENAMES or
                f.startswith("._") or
                f.endswith(".tmp") or
                f.startswith("~")
            )

            # Xóa file rỗng 0-byte (trừ file txt có thể cố ý)
            is_empty_non_txt = (
                os.path.exists(full_path) and
                os.path.getsize(full_path) == 0 and
                not f.endswith(".txt")
            )

            if is_junk or is_empty_non_txt:
                try:
                    os.remove(full_path)
                    removed_count += 1
                except Exception as e:
                    print(f"[Cảnh báo] Không thể xóa file rác {full_path}: {e}")
            else:
                valid_count += 1

        # Xóa thư mục con rỗng
        for d in dirs:
            dir_full_path = os.path.join(root, d)
            if d.startswith(".") or d == "__MACOSX":
                try:
                    shutil.rmtree(dir_full_path, ignore_errors=True)
                    removed_count += 1
                except Exception:
                    pass

    return (removed_count, valid_count)


def get_supported_images(folder_path: str) -> List[str]:
    """Lấy danh sách toàn bộ đường dẫn ảnh hợp lệ trong thư mục."""
    if not os.path.exists(folder_path):
        return []

    images = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if not file.startswith(".") and not file.startswith("._"):
                ext = os.path.splitext(file)[1]
                if ext in IMAGE_EXTENSIONS:
                    images.append(os.path.join(root, file))

    return sorted(images)


def get_supported_videos(folder_path: str) -> List[str]:
    """Lấy danh sách toàn bộ đường dẫn video hợp lệ trong thư mục."""
    if not os.path.exists(folder_path):
        return []

    videos = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if not file.startswith(".") and not file.startswith("._"):
                ext = os.path.splitext(file)[1]
                if ext in VIDEO_EXTENSIONS:
                    videos.append(os.path.join(root, file))

    return sorted(videos)


def validate_image(image_path: str) -> bool:
    """Kiểm tra xem ảnh có đọc được bằng Pillow không và không bị hỏng."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False
