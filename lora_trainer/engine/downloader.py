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
    """Tải file qua aria2c với 16 luồng song song có hỗ trợ Token Header."""
    os.makedirs(destination_dir, exist_ok=True)
    if not filename:
        filename = url.split("/")[-1].split("?")[0]

    dest_path = os.path.join(destination_dir, filename)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0 and not overwrite:
        print(f"✔️ Đã có sẵn file: {filename}")
        return dest_path

    if token is None and "huggingface.co" in url:
        token = get_hf_token()

    if is_aria2_available():
        cmd = [
            "aria2c",
            "--console-log-level=error",
            "-c",
            "-x", "16",
            "-s", "16",
            "-k", "1M",
            "-j", "4",
            "-d", destination_dir,
            "-o", filename,
        ]
        if token and "huggingface.co" in url:
            cmd.append(f"--header=Authorization: Bearer {token}")
        cmd.append(url)
        res = subprocess.run(cmd)
        if res.returncode == 0:
            return dest_path

    # Fallback to requests streaming download
    return download_file_requests(url, dest_path, token=token)


def download_file_requests(url: str, destination_path: str, token: Optional[str] = None) -> str:
    """Tải file bằng thư viện requests có hiển thị tiến trình tqdm."""
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    if token is None and "huggingface.co" in url:
        token = get_hf_token()

    headers = {}
    if token and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(url, headers=headers, stream=True, allow_redirects=True)
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
    filename = os.path.basename(destination_path)
    try:
        return aria2_download(url, dest_dir, filename, overwrite, token=token)
    except Exception as e:
        if fallback_url and ("401" in str(e) or "404" in str(e) or "Unauthorized" in str(e)):
            print(f"⚠️ Link chính gặp lỗi ({e}), chuyển sang link tải dự phòng...")
            return aria2_download(fallback_url, dest_dir, filename, overwrite, token=token)
        raise e


def download_model_suite(
    model_name: str, weights_dir: str = "/content/models", hf_token: Optional[str] = None
) -> Dict[str, str]:
    """
    Tải toàn bộ bộ trọng số cần thiết (Model, VAE, Text Encoder) cho model đã chọn.
    Trả về dictionary chứa đường dẫn local của từng thành phần.
    """
    os.makedirs(weights_dir, exist_ok=True)
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token.strip()

    info = get_model_info(model_name)
    downloaded_paths = {}

    print(f"\n=======================================================")
    print(f"📦 BẮT ĐẦU TẢI TRỌNG SỐ CHO: {model_name}")
    print(f"=======================================================\n")

    # 1. Tải Base DiT Model nếu có download_url
    if "download_url" in info and info["download_url"]:
        url = info["download_url"]
        fb_url = info.get("fallback_url")
        fname = f"{info['arch']}_{model_name.replace(' ', '_')}.safetensors"
        dest = os.path.join(weights_dir, fname)
        print(f"🚀 Đang tải Base DiT Model...")
        downloaded_paths["dit"] = download_file(url, dest, fallback_url=fb_url, token=hf_token)

    # 2. Tải VAE
    if "vae" in info and info["vae"] in VAE_REGISTRY:
        vae_key = info["vae"]
        url = VAE_REGISTRY[vae_key]
        fb_url = VAE_FALLBACKS.get(vae_key)
        ext = ".pth" if "pth" in url else ".safetensors"
        dest = os.path.join(weights_dir, f"{vae_key}{ext}")
        print(f"🚀 Đang tải VAE ({vae_key})...")
        downloaded_paths["vae"] = download_file(url, dest, fallback_url=fb_url, token=hf_token)

    # 3. Tải Text Encoder 1
    if "clip" in info and info["clip"] in TEXT_ENCODER_REGISTRY:
        te_key = info["clip"]
        url = TEXT_ENCODER_REGISTRY[te_key]
        fb_url = TEXT_ENCODER_FALLBACKS.get(te_key)
        ext = ".pth" if "pth" in url else ".safetensors"
        dest = os.path.join(weights_dir, f"{te_key}{ext}")
        print(f"🚀 Đang tải Text Encoder 1 ({te_key})...")
        downloaded_paths["text_encoder1"] = download_file(url, dest, fallback_url=fb_url, token=hf_token)

    # 4. Tải Text Encoder 2 (nếu có)
    if "clip2" in info and info["clip2"] in TEXT_ENCODER_REGISTRY:
        te2_key = info["clip2"]
        url = TEXT_ENCODER_REGISTRY[te2_key]
        fb_url = TEXT_ENCODER_FALLBACKS.get(te2_key)
        ext = ".pth" if "pth" in url else ".safetensors"
        dest = os.path.join(weights_dir, f"{te2_key}{ext}")
        print(f"🚀 Đang tải Text Encoder 2 ({te2_key})...")
        downloaded_paths["text_encoder2"] = download_file(url, dest, fallback_url=fb_url, token=hf_token)

    # 5. Tải Clip Vision (cho I2V)
    if "clip_vision" in info and info["clip_vision"] in TEXT_ENCODER_REGISTRY:
        cv_key = info["clip_vision"]
        url = TEXT_ENCODER_REGISTRY[cv_key]
        ext = ".pth" if "pth" in url else ".safetensors"
        dest = os.path.join(weights_dir, f"{cv_key}{ext}")
        print(f"🚀 Đang tải Clip Vision ({cv_key})...")
        downloaded_paths["clip_vision"] = download_file(url, dest, token=hf_token)

    # 6. Tải Adapter (cho Z-Image Turbo / De-Turbo)
    if "adapter" in info and info["adapter"] in TEXT_ENCODER_REGISTRY:
        ad_key = info["adapter"]
        url = TEXT_ENCODER_REGISTRY[ad_key]
        dest = os.path.join(weights_dir, f"{ad_key}.safetensors")
        print(f"🚀 Đang tải Training Adapter ({ad_key})...")
        downloaded_paths["adapter"] = download_file(url, dest, token=hf_token)

    print("\n✅ Hoàn tất tải toàn bộ trọng số!")
    return downloaded_paths
