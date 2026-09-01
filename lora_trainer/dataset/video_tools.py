"""
Video Preprocessing Tools
Xử lý cắt khung hình (Frame Extraction), tính toán Aspect Ratio Bucketing
và phân tích metadata của video clips dùng cho huấn luyện Video LoRA (Wan 2.1 & 2.2).
"""

import os
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image


def get_video_info(video_path: str) -> Dict[str, Any]:
    """Phân tích thông số kỹ thuật của video clip (FPS, độ phân giải, số khung hình)."""
    info = {"width": 0, "height": 0, "fps": 24.0, "total_frames": 0, "duration": 0.0}
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            info["fps"] = float(cap.get(cv2.CAP_PROP_FPS))
            info["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if info["fps"] > 0:
                info["duration"] = info["total_frames"] / info["fps"]
            cap.release()
    except Exception:
        pass
    return info


def extract_video_frames(
    video_path: str,
    output_dir: str,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    output_format: str = "png",
) -> int:
    """
    Trích xuất từng khung hình của video sang thư mục ảnh riêng biệt.
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        saved_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride == 0:
                frame_filename = os.path.join(output_dir, f"frame_{str(saved_idx).zfill(5)}.{output_format}")
                cv2.imwrite(frame_filename, frame)
                saved_idx += 1
                count += 1
                if max_frames and count >= max_frames:
                    break

            frame_idx += 1

        cap.release()
    except Exception as e:
        print(f"⚠️ Lỗi trích xuất khung hình video {video_path}: {e}")

    return count


def calculate_bucket_resolution(
    width: int,
    height: int,
    max_pixels: int = 1024 * 1024,
    divisible_by: int = 64,
) -> Tuple[int, int]:
    """Tính toán độ phân giải bucket gần nhất chia hết cho 64 hoặc 16 giữ nguyên tỉ lệ khung hình."""
    aspect_ratio = width / height
    target_area = max_pixels

    new_h = int((target_area / aspect_ratio) ** 0.5)
    new_w = int(new_h * aspect_ratio)

    new_w = round(new_w / divisible_by) * divisible_by
    new_h = round(new_h / divisible_by) * divisible_by

    return max(new_w, divisible_by), max(new_h, divisible_by)
