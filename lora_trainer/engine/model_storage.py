"""
Persistent Model Storage & Directory Manager
Quản lý kho lưu trữ model vĩnh viễn trên Google Drive, quét thư mục an toàn không tạo đè,
kiểm tra tính toàn vẹn của tệp và chống tải đè lặp lại.
"""

import os
import shutil
from typing import Dict, Any, List, Optional, Tuple
from ..config.model_registry import get_model_info, VAE_REGISTRY, TEXT_ENCODER_REGISTRY

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/TranningLorasData"
LOCAL_CACHE_DIR = "/content/models"


def setup_storage_structure(base_dir: str = DEFAULT_DRIVE_ROOT) -> Dict[str, str]:
    """
    Khởi tạo cây thư mục chuẩn hóa trên Google Drive an toàn.
    Nếu thư mục đã tồn tại -> BẢO LƯU 100% dữ liệu, KHÔNG tạo đè.
    Nếu chưa có -> Tạo mới.
    """
    folders = {
        "root": base_dir,
        "config": os.path.join(base_dir, "config"),
        "models": os.path.join(base_dir, "models"),
        "models_dit": os.path.join(base_dir, "models", "dit"),
        "models_vae": os.path.join(base_dir, "models", "vae"),
        "models_text_encoders": os.path.join(base_dir, "models", "text_encoders"),
        "datasets": os.path.join(base_dir, "datasets"),
        "train_data": os.path.join(base_dir, "datasets", "train_data"),
        "control_data": os.path.join(base_dir, "datasets", "control_data"),
        "outputs": os.path.join(base_dir, "outputs"),
        "outputs_samples": os.path.join(base_dir, "outputs", "sample_images"),
        "outputs_comfy": os.path.join(base_dir, "outputs", "ComfyUI_Ready"),
        "engines_cache": os.path.join(base_dir, "engines_cache"),
    }

    created_count = 0
    existing_count = 0

    for name, path in folders.items():
        if os.path.exists(path):
            existing_count += 1
        else:
            try:
                os.makedirs(path, exist_ok=True)
                created_count += 1
            except Exception:
                pass

    return folders


def is_file_complete(filepath: str, min_size_bytes: int = 1024 * 1024) -> bool:
    """
    Kiểm tra file đã tải hoàn tất và hợp lệ hay chưa:
    - Tồn tại và dung lượng > min_size_bytes
    - Không có tệp tạm dở dang (.aria2)
    """
    if not filepath or not os.path.exists(filepath):
        return False
    
    aria2_temp = f"{filepath}.aria2"
    if os.path.exists(aria2_temp):
        return False

    try:
        size = os.path.getsize(filepath)
        return size >= min_size_bytes
    except OSError:
        return False


def get_model_component_paths(
    model_name: str,
    base_dir: str = DEFAULT_DRIVE_ROOT,
    local_dir: str = LOCAL_CACHE_DIR,
) -> List[Dict[str, Any]]:
    """
    Trích xuất toàn bộ danh sách các file thành phần cần thiết của một mô hình
    và xác định vị trí lưu trữ tối ưu trên Google Drive / Local SSD.
    """
    info = get_model_info(model_name)
    components = []

    models_dit_dir = os.path.join(base_dir, "models", "dit") if os.path.exists("/content/drive/MyDrive") else local_dir
    models_vae_dir = os.path.join(base_dir, "models", "vae") if os.path.exists("/content/drive/MyDrive") else local_dir
    models_te_dir = os.path.join(base_dir, "models", "text_encoders") if os.path.exists("/content/drive/MyDrive") else local_dir

    # 1. Base DiT
    if "download_url" in info and info["download_url"]:
        fname = f"{info['arch']}_{model_name.replace(' ', '_')}.safetensors"
        drive_path = os.path.join(models_dit_dir, fname)
        local_path = os.path.join(local_dir, fname)
        components.append({
            "type": "DiT Model",
            "key": "dit",
            "name": model_name,
            "filename": fname,
            "url": info["download_url"],
            "fallback_url": info.get("fallback_url"),
            "drive_path": drive_path,
            "local_path": local_path,
        })

    # 2. VAE
    if "vae" in info and info["vae"] in VAE_REGISTRY:
        vae_key = info["vae"]
        url = VAE_REGISTRY[vae_key]
        ext = ".pth" if "pth" in url else ".safetensors"
        fname = f"{vae_key}{ext}"
        drive_path = os.path.join(models_vae_dir, fname)
        local_path = os.path.join(local_dir, fname)
        components.append({
            "type": "VAE",
            "key": "vae",
            "name": vae_key,
            "filename": fname,
            "url": url,
            "drive_path": drive_path,
            "local_path": local_path,
        })

    # 3. Text Encoder 1
    if "clip" in info and info["clip"] in TEXT_ENCODER_REGISTRY:
        te_key = info["clip"]
        url = TEXT_ENCODER_REGISTRY[te_key]
        ext = ".pth" if "pth" in url else ".safetensors"
        fname = f"{te_key}{ext}"
        drive_path = os.path.join(models_te_dir, fname)
        local_path = os.path.join(local_dir, fname)
        components.append({
            "type": "Text Encoder 1",
            "key": "text_encoder1",
            "name": te_key,
            "filename": fname,
            "url": url,
            "drive_path": drive_path,
            "local_path": local_path,
        })

    # 4. Text Encoder 2
    if "clip2" in info and info["clip2"] in TEXT_ENCODER_REGISTRY:
        te2_key = info["clip2"]
        url = TEXT_ENCODER_REGISTRY[te2_key]
        ext = ".pth" if "pth" in url else ".safetensors"
        fname = f"{te2_key}{ext}"
        drive_path = os.path.join(models_te_dir, fname)
        local_path = os.path.join(local_dir, fname)
        components.append({
            "type": "Text Encoder 2",
            "key": "text_encoder2",
            "name": te2_key,
            "filename": fname,
            "url": url,
            "drive_path": drive_path,
            "local_path": local_path,
        })

    # 5. Clip Vision
    if "clip_vision" in info and info["clip_vision"] in TEXT_ENCODER_REGISTRY:
        cv_key = info["clip_vision"]
        url = TEXT_ENCODER_REGISTRY[cv_key]
        ext = ".pth" if "pth" in url else ".safetensors"
        fname = f"{cv_key}{ext}"
        drive_path = os.path.join(models_te_dir, fname)
        local_path = os.path.join(local_dir, fname)
        components.append({
            "type": "Clip Vision",
            "key": "clip_vision",
            "name": cv_key,
            "filename": fname,
            "url": url,
            "drive_path": drive_path,
            "local_path": local_path,
        })

    # 6. Adapter
    if "adapter" in info and info["adapter"] in TEXT_ENCODER_REGISTRY:
        ad_key = info["adapter"]
        url = TEXT_ENCODER_REGISTRY[ad_key]
        fname = f"{ad_key}.safetensors"
        drive_path = os.path.join(models_dit_dir, fname)
        local_path = os.path.join(local_dir, fname)
        components.append({
            "type": "Adapter",
            "key": "adapter",
            "name": ad_key,
            "filename": fname,
            "url": url,
            "drive_path": drive_path,
            "local_path": local_path,
        })

    return components


def scan_model_suite(
    model_name: str,
    base_dir: str = DEFAULT_DRIVE_ROOT,
    local_dir: str = LOCAL_CACHE_DIR,
) -> Dict[str, Any]:
    """
    Quét kiểm tra toàn bộ các file thành phần của mô hình.
    Trả về trạng thái chi tiết của từng tệp: 'ready_drive', 'ready_local', 'missing'.
    """
    components = get_model_component_paths(model_name, base_dir, local_dir)
    results = []
    total_ready_size_bytes = 0
    total_components = len(components)
    ready_components = 0

    for c in components:
        drive_p = c["drive_path"]
        local_p = c["local_path"]

        status = "missing"
        active_path = None
        size_bytes = 0

        if is_file_complete(drive_p):
            status = "ready_drive"
            active_path = drive_p
            size_bytes = os.path.getsize(drive_p)
            ready_components += 1
            total_ready_size_bytes += size_bytes
        elif is_file_complete(local_p):
            status = "ready_local"
            active_path = local_p
            size_bytes = os.path.getsize(local_p)
            ready_components += 1
            total_ready_size_bytes += size_bytes

        size_mb = round(size_bytes / (1024 * 1024), 1)
        size_gb = round(size_bytes / (1024 * 1024 * 1024), 2)

        results.append({
            **c,
            "status": status,
            "active_path": active_path,
            "size_bytes": size_bytes,
            "size_str": f"{size_gb} GB" if size_gb >= 1.0 else f"{size_mb} MB",
        })

    ready_pct = int((ready_components / total_components * 100)) if total_components > 0 else 100

    return {
        "model_name": model_name,
        "components": results,
        "total_components": total_components,
        "ready_components": ready_components,
        "ready_pct": ready_pct,
        "saved_bandwidth_gb": round(total_ready_size_bytes / (1024 * 1024 * 1024), 2),
    }


def display_model_cache_dashboard(
    model_name: str,
    base_dir: str = DEFAULT_DRIVE_ROOT,
    local_dir: str = LOCAL_CACHE_DIR,
) -> None:
    """
    Hiển thị giao diện trực quan tình trạng lưu trữ model trên Google Drive.
    """
    scan = scan_model_suite(model_name, base_dir, local_dir)

    print("\n" + "=" * 70)
    print("📦 BẢNG ĐIỀU KHIỂN MODEL STORAGE (GOOGLE DRIVE PERSISTENT)")
    print(f"📁 Thư mục lưu trữ: {base_dir}/models")
    print(f"🎯 Mô hình kiểm tra: {model_name}")
    print("=" * 70)

    for idx, c in enumerate(scan["components"], 1):
        c_type = c["type"].ljust(14)
        fname = c["filename"].ljust(35)
        
        if c["status"] == "ready_drive":
            stat_str = f"✅ ĐÃ CÓ TRÊN GOOGLE DRIVE ({c['size_str']}) [Không cần tải lại]"
        elif c["status"] == "ready_local":
            stat_str = f"⚡ ĐÃ CÓ TRÊN LOCAL SSD ({c['size_str']})"
        else:
            stat_str = f"⏳ CHƯA CÓ (Sẽ tự động tải và lưu vĩnh viễn vào Drive)"

        print(f"[{idx}] {c_type}: {fname} | {stat_str}")

    print("-" * 70)
    if scan["ready_pct"] == 100:
        print(f"🎉 TRẠNG THÁI: 100% SẴN SÀNG -> Tiết kiệm {scan['saved_bandwidth_gb']} GB băng thông & 0 giây chờ!")
    else:
        print(f"📊 TIẾN ĐỘ: Sẵn sàng {scan['ready_pct']}% ({scan['ready_components']}/{scan['total_components']} thành phần) | Đã lưu sẵn {scan['saved_bandwidth_gb']} GB")
    print("=" * 70 + "\n")
