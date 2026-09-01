"""
High-Speed Model Downloader
Tải xuống các trọng số mô hình (DiT, VAE, Text Encoders) siêu tốc với Aria2c hoặc HuggingFace Hub.
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
from ..config.model_registry import (
    get_model_info,
    VAE_REGISTRY,
    VAE_FALLBACKS,
    TEXT_ENCODER_REGISTRY,
    TEXT_ENCODER_FALLBACKS,
)
from ..caption.key_manager import get_api_key


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
    - Kế thừa 100% logic cốt lõi từ ComfyUI_Model_Downloader.
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
        # Cấu hình lệnh aria2c tối ưu hóa đặc thù cho Google Drive FUSE
        if "huggingface.co" in url or "hf.co" in url:
            aria2_cmd = [
                "aria2c",
                "--console-log-level=error",
                "--summary-interval=1",
                "--file-allocation=none",
                "-c",
                "-x", "4",
                "-s", "4",
                "-k", "10M",
                "-d", destination_dir,
            ]
            if effective_token:
                aria2_cmd.append(f"--header=Authorization: Bearer {effective_token.strip()}")
        else:
            aria2_cmd = [
                "aria2c",
                "--console-log-level=error",
                "--summary-interval=1",
                "--file-allocation=none",
                "-c",
                "-x", "16",
                "-s", "16",
                "-k", "10M",
                "-d", destination_dir,
            ]

        aria2_cmd.extend(["--user-agent=Mozilla/5.0", "--content-disposition"])
        if filename:
            aria2_cmd.extend(["-o", filename])
        aria2_cmd.append(url)

        max_retries = 8
        retry_count = 0
        success = False
        pbar = None

        while retry_count <= max_retries:
            if retry_count > 0:
                print(f"\n⚠️ Kết nối gián đoạn. Đang tự động TẢI TIẾP (Thử lại {retry_count}/{max_retries})...")
                time.sleep(2)

            process = subprocess.Popen(aria2_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                parsed = parse_aria2_progress(line)
                if parsed:
                    if pbar is None:
                        pbar = tqdm(
                            desc=filename,
                            total=parsed["total_bytes"] if parsed["total_bytes"] > 0 else None,
                            unit="iB",
                            unit_scale=True,
                            unit_divisor=1024,
                            leave=True,
                        )
                    elif pbar.total is None or pbar.total <= 0:
                        pbar.total = parsed["total_bytes"]

                    pbar.n = parsed["downloaded_bytes"]
                    pbar.refresh()

            process.wait()

            if process.returncode == 0:
                success = True
                if pbar:
                    if pbar.total and pbar.n < pbar.total:
                        pbar.n = pbar.total
                        pbar.refresh()
                    pbar.close()
                break
            else:
                retry_count += 1

        if pbar and not success:
            pbar.close()

        if success:
            if hasattr(os, "sync"):
                try:
                    os.sync()
                except Exception:
                    pass
            print(f"🎉 Tải xuống thành công! Tệp đã lưu an toàn tại: {dest_path}")
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
        leave=True,
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
    civitai_key: Optional[str] = None,
) -> str:
    """Tải file đơn lẻ đến đích cụ thể, hỗ trợ fallback URL nếu gặp lỗi xác thực 401/404."""
    if os.path.exists(destination_path) and os.path.getsize(destination_path) > 0 and not overwrite:
        return destination_path

    dest_dir = os.path.dirname(destination_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    filename = os.path.basename(destination_path)
    try:
        return aria2_download(url, dest_dir, filename, overwrite, token=token, civitai_key=civitai_key)
    except Exception as e:
        if fallback_url:
            print(f"⚠️ Link chính gặp sự cố ({e}), tự động chuyển sang link tải dự phòng...")
            try:
                return aria2_download(fallback_url, dest_dir, filename, overwrite, token=token, civitai_key=civitai_key)
            except Exception:
                return aria2_download(fallback_url, dest_dir, filename, overwrite, token=None, civitai_key=civitai_key)
        raise e


def download_model_suite(
    model_name: str,
    weights_dir: str = "/content/models",
    hf_token: Optional[str] = None,
    civitai_key: Optional[str] = None,
    base_drive_dir: str = "/content/drive/MyDrive/TranningLorasData",
) -> Dict[str, str]:
    """
    Quản lý và tải toàn bộ bộ trọng số cần thiết (DiT, VAE, Text Encoder).
    Tự động kiểm tra và ưu tiên sử dụng các file đã lưu sẵn trên Google Drive để tránh tải lại.
    Giao diện hiển thị thanh tiến trình chuẩn hóa trực quan và chuyên nghiệp.
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

        # Định dạng nhãn hiển thị theo chuẩn UI
        if "dit" in c_type.lower():
            header_label = "Base DiT Model"
        elif c.get("name"):
            header_label = f"{c_type} ({c['name']})"
        else:
            header_label = f"{c_type} ({fname})"

        # Nếu đã tìm thấy file hợp lệ ở bất kỳ đâu trên Google Drive hoặc Local SSD
        if active_p and is_file_complete(active_p):
            loc_label = "Kho Google Drive" if "drive" in active_p.lower() else "Local SSD"
            print(f"✔️ [{loc_label}] Đã có sẵn {c_type}: {os.path.basename(active_p)} ({c.get('size_str', '')}) -> Bỏ qua tải!")
            downloaded_paths[c_key] = active_p
            continue

        # Chọn đích tải ưu tiên: Google Drive nếu có kết nối, ngược lại là Local SSD
        target_dest = drive_p if os.path.exists("/content/drive/MyDrive") else local_p
        print(f"🚀 Đang tải {header_label}...")
        final_path = download_file(url, target_dest, fallback_url=fallback_url, token=hf_token, civitai_key=civitai_key)
        downloaded_paths[c_key] = final_path

    print("\n✅ Hoàn tất kiểm tra & sẵn sàng toàn bộ trọng số!")
    return downloaded_paths
