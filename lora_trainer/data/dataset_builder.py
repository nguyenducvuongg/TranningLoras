"""
Dataset Builder & Scanner
Phân tích cấu trúc thư mục, trích xuất số bước/repeats từ tên folder và ghép nối tập dữ liệu đa concept / multi-control.
"""

import os
from typing import List, Dict, Any, Tuple, Optional, Union
from PIL import Image
from .cleaner import get_supported_images, get_supported_videos


def parse_folder_steps(
    folder_path: str, default_steps: int = 1000, default_repeats: int = 1
) -> Tuple[int, int, str]:
    """
    Phân tích tên thư mục theo chuẩn `{repeats_or_steps}_{name}`.
    Ví dụ: '100_cyberpunk_girl' -> (100, 100, 'cyberpunk_girl')
    Trả về (steps, repeats, clean_name).
    """
    norm_path = os.path.normpath(folder_path)
    base_name = os.path.basename(norm_path)
    
    if "_" in base_name:
        parts = base_name.split("_", 1)
        if parts[0].isdigit():
            val = int(parts[0])
            name = parts[1].replace(" ", "_")
            return (val, val, name)

    return (default_steps, default_repeats, base_name.replace(" ", "_"))


def build_dataset_list(
    train_folders_input: Union[str, List[str]],
    control_folders_input: Optional[Union[str, List[str]]] = None,
    control_folders_2_input: Optional[Union[str, List[str]]] = None,
    control_folders_3_input: Optional[Union[str, List[str]]] = None,
    default_steps: int = 1000,
    default_repeats: int = 1,
) -> List[Dict[str, Any]]:
    """
    Xây dựng danh sách dataset hoàn chỉnh hỗ trợ đa concept và ghép cặp multi-control.
    """
    # Chuẩn hoá input thành danh sách
    if isinstance(train_folders_input, str):
        train_list = [f.strip() for f in train_folders_input.split(",") if f.strip()]
    else:
        train_list = list(train_folders_input)

    def parse_ctrl_list(ctrl_input: Optional[Union[str, List[str]]]) -> List[str]:
        if not ctrl_input:
            return []
        if isinstance(ctrl_input, str):
            return [f.strip() for f in ctrl_input.split(",") if f.strip()]
        return list(ctrl_input)

    ctrl1_list = parse_ctrl_list(control_folders_input)
    ctrl2_list = parse_ctrl_list(control_folders_2_input)
    ctrl3_list = parse_ctrl_list(control_folders_3_input)

    datasets = []
    for idx, folder in enumerate(train_list):
        if not os.path.exists(folder):
            print(f"[Cảnh báo] Bỏ qua vì không tìm thấy thư mục: {folder}")
            continue

        images = get_supported_images(folder)
        videos = get_supported_videos(folder)
        
        if len(images) == 0 and len(videos) == 0:
            print(f"[Cảnh báo] Thư mục '{folder}' không chứa tệp ảnh/video nào được hỗ trợ.")
            continue

        steps, repeats, name = parse_folder_steps(folder, default_steps, default_repeats)

        c1 = ctrl1_list[idx] if idx < len(ctrl1_list) else None
        c2 = ctrl2_list[idx] if idx < len(ctrl2_list) else None
        c3 = ctrl3_list[idx] if idx < len(ctrl3_list) else None

        ds_item: Dict[str, Any] = {
            "path": folder,
            "name": name,
            "steps": steps,
            "repeats": repeats,
            "num_images": len(images),
            "num_videos": len(videos),
            "control_path": c1 if c1 and os.path.exists(c1) else None,
            "control_path_2": c2 if c2 and os.path.exists(c2) else None,
            "control_path_3": c3 if c3 and os.path.exists(c3) else None,
        }
        datasets.append(ds_item)

    return datasets


def check_folder_stats(folder_path: str) -> Dict[str, Any]:
    """Kiểm tra và thống kê chi tiết tình trạng dữ liệu trong thư mục."""
    if not os.path.exists(folder_path):
        return {"error": f"Thư mục không tồn tại: {folder_path}"}

    images = get_supported_images(folder_path)
    videos = get_supported_videos(folder_path)

    captioned_count = 0
    uncaptioned_count = 0
    resolutions = []

    for img_path in images:
        cap_file = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(cap_file) and os.path.getsize(cap_file) > 0:
            captioned_count += 1
        else:
            uncaptioned_count += 1

        try:
            with Image.open(img_path) as im:
                resolutions.append(im.size)
        except Exception:
            pass

    for vid_path in videos:
        cap_file = os.path.splitext(vid_path)[0] + ".txt"
        if os.path.exists(cap_file) and os.path.getsize(cap_file) > 0:
            captioned_count += 1
        else:
            uncaptioned_count += 1

    stats = {
        "folder": folder_path,
        "total_images": len(images),
        "total_videos": len(videos),
        "captioned_files": captioned_count,
        "uncaptioned_files": uncaptioned_count,
        "caption_ratio_pct": round((captioned_count / max(1, len(images) + len(videos))) * 100, 1),
    }

    if resolutions:
        avg_w = sum(r[0] for r in resolutions) // len(resolutions)
        avg_h = sum(r[1] for r in resolutions) // len(resolutions)
        stats["avg_resolution"] = f"{avg_w}x{avg_h}"

    return stats
