"""
Dataset Cleaner & Validator
Quét sạch các tệp rác (.DS_Store, tệp tạm), kiểm tra tính toàn vẹn của ảnh/video,
phát hiện và loại bỏ các file bị lỗi hỏng (corrupt).
"""

import os
from typing import List
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".MP4", ".MKV", ".MOV", ".AVI", ".WEBM"}


def get_supported_images(dir_path: str) -> List[str]:
    """Lấy danh sách tất cả các ảnh hợp lệ trong thư mục."""
    if not os.path.exists(dir_path):
        return []
    valid_files = []
    for f in os.listdir(dir_path):
        ext = os.path.splitext(f)[1]
        if ext in IMAGE_EXTENSIONS:
            valid_files.append(os.path.join(dir_path, f))
    return sorted(valid_files)


def get_supported_videos(dir_path: str) -> List[str]:
    """Lấy danh sách tất cả các video hợp lệ trong thư mục."""
    if not os.path.exists(dir_path):
        return []
    valid_files = []
    for f in os.listdir(dir_path):
        ext = os.path.splitext(f)[1]
        if ext in VIDEO_EXTENSIONS:
            valid_files.append(os.path.join(dir_path, f))
    return sorted(valid_files)


def clean_directory(dir_path: str, remove_corrupted: bool = True) -> int:
    """
    Dọn dẹp thư mục huấn luyện:
    - Xóa các tệp hệ thống .DS_Store, Thumbs.db, ._*, v.v.
    - Kiểm tra và xóa các ảnh bị lỗi/hỏng không thể nạp bằng PIL.
    """
    if not os.path.exists(dir_path):
        print(f"⚠️ Thư mục không tồn tại: {dir_path}")
        return 0

    removed_count = 0
    corrupted_count = 0

    for root, _, files in os.walk(dir_path):
        for f in files:
            fpath = os.path.join(root, f)
            # 1. Xóa file rác
            if f.startswith("._") or f in [".DS_Store", "Thumbs.db", "desktop.ini"]:
                try:
                    os.remove(fpath)
                    removed_count += 1
                except Exception:
                    pass
                continue

            # 2. Kiểm tra ảnh hỏng
            ext = os.path.splitext(f)[1]
            if ext in IMAGE_EXTENSIONS and remove_corrupted:
                try:
                    with Image.open(fpath) as img:
                        img.verify()
                except Exception:
                    print(f"❌ Phát hiện và xóa ảnh bị lỗi: {f}")
                    try:
                        os.remove(fpath)
                        txt_path = os.path.splitext(fpath)[0] + ".txt"
                        if os.path.exists(txt_path):
                            os.remove(txt_path)
                        corrupted_count += 1
                    except Exception:
                        pass

    print(f"🧹 Đã dọn dẹp thư mục: {dir_path} (Đã xóa {removed_count} file rác, {corrupted_count} file hỏng).")
    return removed_count + corrupted_count
