"""
High-Speed Model Downloader
Tải xuống các trọng số mô hình (DiT, VAE, Text Encoders) siêu tốc với Aria2c hoặc HuggingFace Hub.
"""

import os
import shutil
import subprocess
from typing import Dict, Any, List, Optional
import requests
from tqdm import tqdm
from ..config.model_registry import (
    get_model_info,
    VAE_REGISTRY,
    VAE_FALLBACKS,
    TEXT_ENCODER_REGISTRY,
    TEXT_ENCODER_FALLBACKS,
)
from ..caption.key_manager import get_api_key


def is_aria2_available() -> bool:
    """Kiểm tra aria2c có sẵn trong hệ thống hay không."""
    return shutil.which("aria2c") is not None


def get_hf_token() -> Optional[str]:
    """Lấy Hugging Face token từ key manager hoặc biến môi trường."""
    return get_api_key("huggingface") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def aria2_download(
    url: str,
    destination_dir: str,
    filename: Optional[str] = None,
    overwrite: bool = False,
    token: Optional[str] = None,
) -> str:
    """Tải file qua aria2c với 16 luồng song song, tối ưu I/O và tự động fallback."""
    os.makedirs(destination_dir, exist_ok=True)
    if not filename:
        filename = url.split("/")[-1].split("?")[0]

    dest_path = os.path.join(destination_dir, filename)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0 and not overwrite:
        print(f"✔️ Đã có sẵn file: {filename}")
        return dest_path

    effective_token = token if token is not None else get_hf_token()

    if is_aria2_available():
        cmd = [
            "aria2c",
            "--console-log-level=error",
            "-c",
            "-x", "16",
            "-s", "16",
            "-k", "1M",
            "-j", "8",
            "--file-allocation=none",
            "--disk-cache=64M",
            "--optimize-concurrent-downloads=true",
            "--summary-interval=5",
            "-d", destination_dir,
            "-o", filename,
        ]
        if effective_token and ("huggingface.co" in url or "hf.co" in url):
            cmd.append(f"--header=Authorization: Bearer {effective_token.strip()}")
        cmd.append(url)
        res = subprocess.run(cmd)
        if res.returncode == 0:
            return dest_path

        # Nếu lỗi và có token, thử lại aria2c không dùng token (cho public repos)
        if effective_token:
            cmd_no_token = [
                "aria2c",
                "--console-log-level=error",
                "-c",
                "-x", "16",
                "-s", "16",
                "-k", "1M",
                "-j", "8",
                "--file-allocation=none",
                "--disk-cache=64M",
                "--optimize-concurrent-downloads=true",
                "--summary-interval=5",
                "-d", destination_dir,
                "-o", filename,
                url,
            ]
            res_anon = subprocess.run(cmd_no_token)
            if res_anon.returncode == 0:
                return dest_path

    # Fallback sang requests streaming download
    return download_file_requests(url, dest_path, token=effective_token)


def download_file_requests(url: str, destination_path: str, token: Optional[str] = None) -> str:
    """Tải file bằng thư viện requests có hiển thị tiến trình tqdm và tự phục hồi khi lỗi token."""
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    effective_token = token if token is not None else get_hf_token()

    headers = {}
    if effective_token and ("huggingface.co" in url or "hf.co" in url):
        headers["Authorization"] = f"Bearer {effective_token.strip()}"

    response = requests.get(url, headers=headers, stream=True, allow_redirects=True)
    
    # Nếu token bị 401/403, tự động thử lại yêu cầu ẩn danh không token
    if response.status_code in (401, 403) and effective_token:
        response = requests.get(url, stream=True, allow_redirects=True)

    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    desc = os.path.basename(destination_path)

    with open(destination_path, "wb") as f, tqdm(
        desc=desc,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024 * 1024):
            size = f.write(data)
            pbar.update(size)

    return destination_path


def download_file(
    url: str,
    destination_path: str,
    overwrite: bool = False,
    fallback_url: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    """Tải file đơn lẻ đến đích cụ thể, hỗ trợ fallback URL nếu gặp lỗi xác thực 401/404."""
    if os.path.exists(destination_path) and os.path.getsize(destination_path) > 0 and not overwrite:
        return destination_path

    dest_dir = os.path.dirname(destination_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    filename = os.path.basename(destination_path)
    try:
        return aria2_download(url, dest_dir, filename, overwrite, token=token)
    except Exception as e:
        if fallback_url:
            print(f"⚠️ Link chính gặp sự cố ({e}), tự động chuyển sang link tải dự phòng...")
            try:
                return aria2_download(fallback_url, dest_dir, filename, overwrite, token=token)
            except Exception:
                return aria2_download(fallback_url, dest_dir, filename, overwrite, token=None)
        raise e


def download_model_suite(
    model_name: str,
    weights_dir: str = "/content/models",
    hf_token: Optional[str] = None,
    base_drive_dir: str = "/content/drive/MyDrive/TranningLorasData",
) -> Dict[str, str]:
    """
    Quản lý và tải toàn bộ bộ trọng số cần thiết (DiT, VAE, Text Encoder).
    Tự động kiểm tra và ưu tiên sử dụng các file đã lưu sẵn trên Google Drive để tránh tải lại.
    """
    os.makedirs(weights_dir, exist_ok=True)
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token.strip()

    from .model_storage import scan_model_suite, display_model_cache_dashboard, is_file_complete

    # 1. Hiển thị bảng điều khiển trạng thái các file
    display_model_cache_dashboard(model_name, base_drive_dir, weights_dir)
    scan = scan_model_suite(model_name, base_drive_dir, weights_dir)

    downloaded_paths = {}

    for c in scan["components"]:
        c_key = c["key"]
        c_type = c["type"]
        fname = c["filename"]
        url = c["url"]
        fallback_url = c.get("fallback_url")
        drive_p = c["drive_path"]
        local_p = c["local_path"]
        active_p = c.get("active_path")

        # Nếu đã tìm thấy file hợp lệ ở bất kỳ đâu trên Google Drive hoặc Local SSD
        if active_p and is_file_complete(active_p):
            loc_label = "Kho Google Drive" if "drive" in active_p.lower() else "Local SSD"
            print(f"✔️ [{loc_label}] Đã có sẵn {c_type}: {os.path.basename(active_p)} ({c.get('size_str', '')}) -> Bỏ qua tải!")
            downloaded_paths[c_key] = active_p
            continue

        # Chọn đích tải ưu tiên: Google Drive nếu có kết nối, ngược lại là Local SSD
        target_dest = drive_p if os.path.exists("/content/drive/MyDrive") else local_p
        print(f"🚀 Đang tải {c_type} ({fname})...")
        final_path = download_file(url, target_dest, fallback_url=fallback_url, token=hf_token)
        downloaded_paths[c_key] = final_path

    print("\n✅ Hoàn tất kiểm tra & sẵn sàng toàn bộ trọng số!")
    return downloaded_paths
