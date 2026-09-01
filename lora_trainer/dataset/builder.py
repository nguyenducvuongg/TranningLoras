"""
Dataset Structure Builder & Repeater Parser
Phân tích cú pháp tên thư mục (ví dụ '20_character' -> repeats = 20),
tự động phát hiện cặp đối chiếu Paired Control và sinh danh sách metadata Dataset chuẩn.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from .cleaner import get_supported_images, get_supported_videos


def parse_folder_steps(folder_path: str) -> Tuple[int, str]:
    """
    Phân tích tên thư mục theo quy chuẩn Kohya:
    '20_girl' -> repeats = 20, concept = 'girl'
    'my_style' -> repeats = 1, concept = 'my_style'
    """
    folder_name = os.path.basename(os.path.normpath(folder_path))
    parts = folder_name.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return int(parts[0]), parts[1]
    return 1, folder_name


def check_folder_stats(folder_path: str) -> Dict[str, Any]:
    """Kiểm tra số lượng ảnh, video và file caption trong một thư mục."""
    imgs = get_supported_images(folder_path)
    vids = get_supported_videos(folder_path)
    txts = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".txt")] if os.path.exists(folder_path) else []

    repeats, concept = parse_folder_steps(folder_path)
    return {
        "path": folder_path,
        "image_count": len(imgs),
        "video_count": len(vids),
        "caption_count": len(txts),
        "repeats": repeats,
        "concept": concept,
    }


def build_dataset_list(
    train_folders: str,
    control_folder: Optional[str] = None,
    resolution: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Xây dựng danh sách cấu hình Dataset từ chuỗi đường dẫn (phân tách bởi dấu phẩy).
    Tự động ghép cặp Control Folder nếu có.
    """
    raw_dirs = [d.strip() for d in train_folders.split(",") if d.strip()]
    datasets = []

    ctrl_dir = control_folder.strip() if control_folder and control_folder.strip() else None

    for d in raw_dirs:
        if not os.path.exists(d):
            print(f"⚠️ Cảnh báo: Thư mục không tồn tại: {d}")
            continue

        stats = check_folder_stats(d)
        item: Dict[str, Any] = {
            "path": d,
            "repeats": stats["repeats"],
            "concept": stats["concept"],
            "image_count": stats["image_count"],
            "video_count": stats["video_count"],
        }
        if resolution:
            item["resolution"] = resolution

        if ctrl_dir and os.path.exists(ctrl_dir):
            item["control_path"] = ctrl_dir

        datasets.append(item)

    return datasets
