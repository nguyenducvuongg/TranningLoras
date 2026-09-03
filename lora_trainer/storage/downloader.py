"""
High-Speed Model Downloader
Tải xuống các trọng số mô hình (Base Model, DiT, UNet, VAE, Text Encoders) siêu tốc với Aria2c hoặc HuggingFace Hub.
Hỗ trợ thanh tiến trình tqdm mượt mà, tự động phục hồi kết nối, thử lại 8 lần và cơ chế Anonymous Fallback khi lỗi 401.
"""

import os
import re
import shutil
import subprocess
import time
import posixpath
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse, parse_qs, urlunparse, urlencode
import requests
from tqdm import tqdm
from ..core.key_vault import get_api_key
from ..core.model_registry import (
    get_model_info,
    VAE_REGISTRY,
    VAE_FALLBACKS,
    TEXT_ENCODER_REGISTRY,
    TEXT_ENCODER_FALLBACKS,
)


def parse_aria2_size(size_str: str) -> int:
    """Chuyển đổi chuỗi dung lượng (ví dụ '16.2GiB', '320MiB', '500KiB') thành số bytes."""
    size_str = size_str.strip()
    match = re.match(r"^([\d\.]+)\s*([KMGTPE]?i?B?)$", size_str, re.IGNORECASE)
    if not match:
        return 0
    val = float(match.group(1))
    unit = match.group(2).upper()
    multiplier = 1
    if "T" in unit:
        multiplier = 1024 ** 4
    elif "G" in unit:
        multiplier = 1024 ** 3
    elif "M" in unit:
        multiplier = 1024 ** 2
    elif "K" in unit:
        multiplier = 1024
    return int(val * multiplier)


def parse_aria2_progress(line: str) -> Optional[Dict[str, Any]]:
    """Phân tích dòng tiến trình của aria2c để cập nhật thanh tqdm chuẩn đẹp."""
    pattern = r"\[#\w+\s+([\d\.]+\w+)/([\d\.]+\w+)\((\d+)%\).*?DL:([\d\.]+\w+)"
    m = re.search(pattern, line)
    if m:
        downloaded_str = m.group(1)
        total_str = m.group(2)
        pct = int(m.group(3))
        speed_str = m.group(4)
        return {
            "downloaded_bytes": parse_aria2_size(downloaded_str),
            "total_bytes": parse_aria2_size(total_str),
            "percent": pct,
            "speed": speed_str,
        }
    return None


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
            subprocess.run("apt-get update -qq && apt-get install -y -qq aria2", shell=True, check=False)
            return is_aria2_available()
        except Exception:
            pass
    return False


def get_hf_token() -> Optional[str]:
    """Lấy Hugging Face token từ key manager hoặc biến môi trường."""
    return get_api_key("huggingface") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def get_civitai_key() -> Optional[str]:
    """Lấy Civitai API Key từ key manager hoặc biến môi trường."""
    return get_api_key("civitai") or os.environ.get("CIVITAI_API_KEY")


def prepare_download_url(url: str, token: Optional[str] = None, civitai_key: Optional[str] = None) -> str:
    """
    Chuẩn hóa và tối ưu hóa URL tải:
    - Chuyển đổi blob sang resolve cho Hugging Face.
    - Nhúng Civitai API Token vào URL download của Civitai.
    """
    url = url.strip()
    if "huggingface.co" in url and "/blob/main/" in url:
        url = url.replace("/blob/main/", "/resolve/main/")

    active_civitai = civitai_key or get_civitai_key()
    if "civitai.com/api/download/models" in url and active_civitai:
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        query_params["token"] = [active_civitai.strip()]
        url = urlunparse(parsed_url._replace(query=urlencode(query_params, doseq=True)))

    return url


def aria2_download(
    url: str,
    destination_dir: str,
    filename: Optional[str] = None,
    overwrite: bool = False,
    token: Optional[str] = None,
    civitai_key: Optional[str] = None,
) -> str:
    """
    Tải file mô hình trực tiếp vào Google Drive bằng aria2c:
    - Giao diện thanh tiến trình chuẩn hóa tqdm (hiển thị % dung lượng, thời gian, tốc độ).
    - Đa luồng siêu tốc với --file-allocation=none (tương thích tuyệt đối với Google Drive FUSE).
    - Tự động resume / nối tiếp khi mạng chập chờn với vòng lặp thử lại thông minh (max 8 lần).
    """
    os.makedirs(destination_dir, exist_ok=True)
    url = prepare_download_url(url, token=token, civitai_key=civitai_key)

    if not filename:
        parsed_path = urlparse(url.split("?")[0]).path
        basename = posixpath.basename(parsed_path)
        valid_exts = ["safetensors", "ckpt", "pt", "pth", "bin", "onnx", "yaml", "json", "gguf"]
        if "." in basename and basename.split(".")[-1].lower() in valid_exts:
            filename = basename
        else:
            filename = url.split("/")[-1].split("?")[0]

    dest_path = os.path.join(destination_dir, filename)
    aria2_control_file = dest_path + ".aria2"

    # Kiểm tra nếu file đã có sẵn hoàn chỉnh và không có file tạm dở dang (.aria2)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0 and not os.path.exists(aria2_control_file) and not overwrite:
        print(f"✔️ Đã có sẵn file: {filename}")
        return dest_path

    effective_token = token if token is not None else get_hf_token()

    if ensure_aria2_installed():
        cmd = [
            "aria2c",
            "--console-log-level=warn",
            "--summary-interval=1",
            "--file-allocation=none",
            "-x", "16",
            "-s", "16",
            "-j", "16",
            "-k", "1M",
            "-c",
            "--dir", destination_dir,
            "-o", filename,
        ]

        if "huggingface.co" in url and effective_token:
            cmd.extend(["--header", f"Authorization: Bearer {effective_token}"])

        cmd.append(url)

        max_retries = 8
        attempt = 0
        while attempt < max_retries:
            attempt += 1
            print(f"🚀 [Aria2c] Đang tải {filename} (Lần thử {attempt}/{max_retries})...")
            
            pbar = tqdm(
                total=100,
                unit="%",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}, {postfix}]",
                desc=f"📥 {filename[:30]}",
                leave=True,
            )
            
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                )

                last_pct = 0
                error_occurred = False
                unauthorized_error = False

                for line in process.stdout:
                    info = parse_aria2_progress(line)
                    if info:
                        pct = info["percent"]
                        if pct > last_pct:
                            pbar.update(pct - last_pct)
                            last_pct = pct
                        pbar.set_postfix_str(f"{info['speed']}")
                    
                    if "status=401" in line or "Unauthorized" in line:
                        unauthorized_error = True
                    if "ERROR" in line or "errorCode=" in line:
                        error_occurred = True

                process.wait()
                if last_pct < 100 and process.returncode == 0:
                    pbar.update(100 - last_pct)
                pbar.close()

                if process.returncode == 0 and os.path.exists(dest_path) and not os.path.exists(aria2_control_file):
                    print(f"✅ Tải thành công: {dest_path}")
                    return dest_path
                else:
                    if unauthorized_error and effective_token:
                        print(f"⚠️ Phát hiện lỗi 401 Unauthorized với HF Token! Đang chuyển sang chế độ tải công khai...")
                        effective_token = None
                        cmd = [c for c in cmd if not (isinstance(c, str) and ("Authorization" in c or "Bearer" in c))]
                        if "--header" in cmd:
                            cmd.remove("--header")
                    print(f"⚠️ Aria2c gặp sự cố (mã {process.returncode}), đang tự động nối lại tiến trình...")
                    time.sleep(3)
            except Exception as e:
                pbar.close()
                print(f"⚠️ Lỗi thực thi Aria2c: {e}")
                time.sleep(3)

    # Fallback bằng Requests nếu Aria2c không thành công
    print(f"🔄 Đang chuyển sang phương thức tải Requests fallback...")
    return requests_fallback_download(url, dest_path, effective_token)


def requests_fallback_download(url: str, dest_path: str, token: Optional[str] = None) -> str:
    """Tải xuống tệp với thanh tiến trình tqdm sử dụng thư viện Requests tiêu chuẩn."""
    headers = {}
    if "huggingface.co" in url and token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(url, headers=headers, stream=True, timeout=60)
    if response.status_code == 401 and token:
        print(f"⚠️ Lỗi 401 Unauthorized khi dùng Token HF, đang thử tải ẩn danh không token...")
        response = requests.get(url, stream=True, timeout=60)

    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"📥 {os.path.basename(dest_path)[:30]}",
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192 * 16):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"✅ Tải thành công: {dest_path}")
    aria2_temp = dest_path + ".aria2"
    if os.path.exists(aria2_temp):
        try:
            os.remove(aria2_temp)
        except Exception:
            pass
    return dest_path
