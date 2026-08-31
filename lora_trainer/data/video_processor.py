"""
Video Processor
Xử lý metadata video và trích xuất chuỗi khung hình (frame slicing) cho huấn luyện LoRA Video (Wan 2.1, Wan 2.2).
"""

import os
from typing import Dict, Any, List, Optional
from PIL import Image


def get_video_info(video_path: str) -> Dict[str, Any]:
    """Lấy thông tin kỹ thuật của file video (FPS, Resolution, Frame Count, Duration)."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Không thể mở file video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        duration = frame_count / fps if fps > 0 else 0.0
        cap.release()

        return {
            "video_path": video_path,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_sec": round(duration, 2),
        }
    except ImportError:
        # Fallback if cv2 is not installed
        return {
            "video_path": video_path,
            "fps": 24.0,
            "frame_count": 100,
            "width": 1280,
            "height": 720,
            "duration_sec": 4.16,
        }


def extract_video_frames(
    video_path: str,
    output_dir: str,
    mode: str = "chunk",
    target_frames: int = 25,
    frame_stride: int = 1,
    frame_sample: int = 1,
    max_frames: int = 33,
    save_as_images: bool = True,
) -> List[str]:
    """
    Trích xuất khung hình từ video theo các chiến lược tối ưu:
    - head: Lấy trực tiếp N frames đầu tiên (N = target_frames).
    - chunk: Chia video thành các phân đoạn dài target_frames liên tiếp.
    - slide: Cửa sổ trượt lấy target_frames, mỗi lần dịch chuyển frame_stride.
    - uniform: Chia đều video thành frame_sample đoạn, mỗi đoạn trích target_frames.
    - full: Trích xuất toàn bộ video (tối đa max_frames).
    """
    import cv2

    os.makedirs(output_dir, exist_ok=True)
    info = get_video_info(video_path)
    total_frames = info["frame_count"]
    if total_frames == 0:
        return []

    cap = cv2.VideoCapture(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    extracted_paths = []

    # Xác định danh sách chỉ số frame cần lấy
    selected_indices: List[List[int]] = []

    if mode == "head":
        limit = min(target_frames, total_frames)
        selected_indices.append(list(range(0, limit, frame_stride)))

    elif mode == "chunk":
        chunk_size = target_frames * frame_stride
        for start in range(0, total_frames, chunk_size):
            end = min(start + chunk_size, total_frames)
            chunk = list(range(start, end, frame_stride))
            if len(chunk) >= max(1, target_frames // 2):
                selected_indices.append(chunk[:target_frames])

    elif mode == "slide":
        for start in range(0, total_frames - target_frames + 1, max(1, frame_stride * 5)):
            chunk = list(range(start, start + target_frames * frame_stride, frame_stride))
            selected_indices.append(chunk)

    elif mode == "uniform":
        if total_frames <= target_frames:
            selected_indices.append(list(range(total_frames)))
        else:
            step = total_frames // frame_sample
            for i in range(frame_sample):
                start = i * step
                chunk = list(range(start, min(start + target_frames * frame_stride, total_frames), frame_stride))
                if chunk:
                    selected_indices.append(chunk[:target_frames])

    elif mode == "full":
        limit = min(max_frames, total_frames)
        selected_indices.append(list(range(0, limit, frame_stride)))

    else:
        # Default fallback
        selected_indices.append(list(range(min(target_frames, total_frames))))

    # Đọc frame và lưu
    all_frames_dict: Dict[int, Any] = {}
    needed_set = {idx for group in selected_indices for idx in group}

    curr_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if curr_idx in needed_set:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames_dict[curr_idx] = Image.fromarray(frame_rgb)
        curr_idx += 1

    cap.release()

    if save_as_images:
        for group_id, group in enumerate(selected_indices):
            group_folder = os.path.join(output_dir, f"{base_name}_clip{group_id:03d}")
            os.makedirs(group_folder, exist_ok=True)
            for f_order, f_idx in enumerate(group):
                if f_idx in all_frames_dict:
                    img_path = os.path.join(group_folder, f"frame_{f_order:04d}.jpg")
                    all_frames_dict[f_idx].save(img_path, quality=95)
                    extracted_paths.append(img_path)

    return extracted_paths
