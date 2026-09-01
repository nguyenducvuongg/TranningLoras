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


import time


def is_aria2_available() -> bool:
    """Kiểm tra aria2c có sẵn trong hệ thống hay không."""
    return shutil.which("aria2c") is not None


def ensure_aria2_installed() -> bool:
    """Tự động kiểm tra và cài đặt aria2 nếu chạy trên môi trường Linux / Colab."""
    if is_aria2_available():
        return True
    if shutil.which("apt-get"):
        try:
            print("📦 Đang tự động cài đặt aria2 siêu tốc cho Colab...")
            subprocess.run(["apt-get", "install", "-y", "-qq", "aria2"], check=False)
            return is_aria2_available()
        except Exception:
            pass
    return False


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
    """Tải file qua aria2c với 16 luồng song song, tối ưu I/O, tự nối tiếp (resume) và tự động fallback."""
    os.makedirs(destination_dir, exist_ok=True)
    if not filename:
        filename = url.split("/")[-1].split("?")[0]

    dest_path = os.path.join(destination_dir, filename)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0 and not overwrite:
        print(f"✔️ Đã có sẵn file: {filename}")
        return dest_path

    effective_token = token if token is not None else get_hf_token()

    if ensure_aria2_installed():
        cmd = [
            "aria2c",
            "--console-log-level=error",
            "-c",
            "-x", "16",
            "-s", "16",
            "-k", "1M",
            "-j", "8",
            "--max-tries=10",
            "--retry-wait=3",
            "--timeout=60",
            "--connect-timeout=30",
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
                "--max-tries=10",
                "--retry-wait=3",
                "--timeout=60",
                "--connect-timeout=30",
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

    # Fallback sang requests streaming download với cơ chế nối tiếp (HTTP Range Resume)
    return download_file_requests(url, dest_path, token=effective_token)


def download_file_requests(
    url: str,
    destination_path: str,
    token: Optional[str] = None,
    max_retries: int = 15,
) -> str:
    """
    Tải file bằng thư viện requests có hiển thị tiến trình tqdm và tự phục hồi, nối tiếp (HTTP Range)
    khi gặp sự cố IncompleteRead / ngắt kết nối mạng.
    """
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    effective_token = token if token is not None else get_hf_token()

    temp_path = destination_path + ".part"
    existing_bytes = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0

    headers = {}
    if effective_token and ("huggingface.co" in url or "hf.co" in url):
        headers["Authorization"] = f"Bearer {effective_token.strip()}"

    total_size = 0
    try:
        head_resp = requests.head(url, headers=headers, allow_redirects=True, timeout=30)
        total_size = int(head_resp.headers.get("content-length", 0))
    except Exception:
        pass

    desc = os.path.basename(destination_path)

    with tqdm(
        desc=desc,
        total=total_size if total_size > 0 else None,
        initial=existing_bytes,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for attempt in range(max_retries):
            existing_bytes = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
            if total_size > 0 and existing_bytes >= total_size:
                break

            req_headers = dict(headers)
            if existing_bytes > 0:
                req_headers["Range"] = f"bytes={existing_bytes}-"

            try:
                with requests.get(url, headers=req_headers, stream=True, timeout=60, allow_redirects=True) as response:
                    # Nếu token bị lỗi 401/403, thử tải không token
                    if response.status_code in (401, 403) and effective_token:
                        req_headers.pop("Authorization", None)
                        with requests.get(url, headers=req_headers, stream=True, timeout=60, allow_redirects=True) as r_anon:
                            r_anon.raise_for_status()
                            with open(temp_path, "ab" if existing_bytes > 0 else "wb") as f:
                                for chunk in r_anon.iter_content(chunk_size=2 * 1024 * 1024):
                                    if chunk:
                                        f.write(chunk)
                                        pbar.update(len(chunk))
                    else:
                        response.raise_for_status()
                        with open(temp_path, "ab" if existing_bytes > 0 else "wb") as f:
                            for chunk in response.iter_content(chunk_size=2 * 1024 * 1024):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))

                # Kiểm tra hoàn tất
                curr_size = os.path.getsize(temp_path)
                if total_size == 0 or curr_size >= total_size:
                    break

            except Exception as e:
                curr_size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
                print(f"\n⚠️ Mạng gián đoạn tại {round(curr_size / (1024**3), 2)} GB ({e}). Đang tự động nối tiếp tải sau 3s (lần thử {attempt + 1}/{max_retries})...")
                time.sleep(3)

    if os.path.exists(temp_path):
        if os.path.exists(destination_path):
            os.remove(destination_path)
        os.rename(temp_path, destination_path)

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
