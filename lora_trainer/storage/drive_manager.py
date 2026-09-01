"""
Google Drive Persistent Storage Manager
Quản lý cây thư mục chuẩn hóa trên Google Drive, kiểm tra an toàn bảo lưu 100% dữ liệu,
chống ghi đè và xác minh tính toàn vẹn của tệp mô hình đã lưu.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from ..core.model_registry import get_model_info, VAE_REGISTRY, TEXT_ENCODER_REGISTRY

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/TranningLorasData"
LOCAL_CACHE_DIR = "/content/models"


def setup_storage_structure(base_dir: str = DEFAULT_DRIVE_ROOT) -> Dict[str, str]:
    """
    Khởi tạo cây thư mục chuẩn hóa trên Google Drive an toàn.
    Nếu thư mục đã tồn tại -> BẢO LƯU 100% dữ liệu, KHÔNG tạo đè hay xóa file.
    Nếu chưa có -> Tạo mới.
    """
    folders = {
        "root": base_dir,
        "config": os.path.join(base_dir, "config"),
        "models": os.path.join(base_dir, "models"),
        "models_dit": os.path.join(base_dir, "models", "dit"),
        "models_vae": os.path.join(base_dir, "models", "vae"),
        "models_text_encoders": os.path.join(base_dir, "models", "text_encoders"),
        "models_sdxl": os.path.join(base_dir, "models", "sdxl"),
        "models_sd15": os.path.join(base_dir, "models", "sd15"),
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

    if created_count > 0:
        print(f"📁 Đã tạo mới {created_count} thư mục cấu trúc trên Google Drive: {base_dir}")
    if existing_count > 0:
        print(f"🛡️ Đã quét an toàn & bảo lưu nguyên vẹn dữ liệu tại: {base_dir}")

    return folders


def is_file_complete(filepath: str, min_size_bytes: int = 1024 * 1024) -> bool:
    """
    Kiểm tra file đã tải hoàn tất và hợp lệ:
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


def find_existing_file_across_storage(
    candidate_filenames: List[str],
    search_dirs: List[str],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Tìm kiếm file khả dĩ trong tất cả các thư mục tiềm năng trên Google Drive & Local SSD.
    Trả về (đường_dẫn_tồn_tại, loại_vị_trí: 'ready_drive' | 'ready_local').
    """
    clean_fnames = list(dict.fromkeys([f for f in candidate_filenames if f and f.strip()]))
    clean_dirs = list(dict.fromkeys([d for d in search_dirs if d and os.path.exists(d)]))

    for d in clean_dirs:
        for fname in clean_fnames:
            candidate_path = os.path.join(d, fname)
            if is_file_complete(candidate_path):
                location_type = "ready_drive" if "drive" in candidate_path.lower() else "ready_local"
                return candidate_path, location_type

    return None, None


def get_component_target_subfolder(component_type: str) -> str:
    """Xác định thư mục con phù hợp cho từng loại thành phần mô hình."""
    c = component_type.lower().strip()
    if c == "vae":
        return "vae"
    elif "clip" in c or "t5" in c or "encoder" in c:
        return "text_encoders"
    elif "sdxl" in c:
        return "sdxl"
    elif "sd15" in c or "sd1.5" in c:
        return "sd15"
    else:
        return "dit"


def get_model_component_paths(
    model_name: str,
    base_dir: str = DEFAULT_DRIVE_ROOT,
    local_dir: str = LOCAL_CACHE_DIR,
    custom_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Xác định toàn bộ danh sách các file mô hình cần thiết, vị trí lưu trữ trên Google Drive
    và trạng thái tải xuống hiện tại.
    """
    info = get_model_info(model_name, custom_download_url=custom_url)
    components = []

    # 1. Base Model / DiT / UNet
    dit_url = info.get("download_url") or custom_url
    if dit_url:
        raw_fname = dit_url.split("/")[-1].split("?")[0]
        valid_exts = [".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".onnx", ".gguf"]
        if any(raw_fname.lower().endswith(ext) for ext in valid_exts):
            fname = raw_fname
        else:
            clean_name = model_name.replace(" ", "_").replace("/", "_").replace(":", "_")
            fname = f"{clean_name}.safetensors"

        sub = get_component_target_subfolder(info.get("arch", "dit"))
        drive_path = os.path.join(base_dir, "models", sub, fname)
        local_path = os.path.join(local_dir, fname)

        existing_path, location_type = find_existing_file_across_storage(
            [fname],
            [
                os.path.join(base_dir, "models", sub),
                os.path.join(base_dir, "models"),
                local_dir,
            ],
        )

        components.append({
            "name": f"Base Model ({info.get('arch', 'Model').upper()})",
            "type": "dit",
            "filename": fname,
            "url": dit_url,
            "fallback_url": info.get("fallback_url"),
            "drive_path": drive_path,
            "local_path": local_path,
            "active_path": existing_path or drive_path,
            "status": location_type or "needs_download",
        })

    # 2. VAE
    vae_key = info.get("vae")
    if vae_key and vae_key in VAE_REGISTRY:
        vae_url = VAE_REGISTRY[vae_key]
        fname = vae_url.split("/")[-1].split("?")[0]
        drive_path = os.path.join(base_dir, "models", "vae", fname)
        local_path = os.path.join(local_dir, fname)

        existing_path, location_type = find_existing_file_across_storage(
            [fname],
            [
                os.path.join(base_dir, "models", "vae"),
                os.path.join(base_dir, "models"),
                local_dir,
            ],
        )

        components.append({
            "name": f"VAE ({vae_key})",
            "type": "vae",
            "filename": fname,
            "url": vae_url,
            "drive_path": drive_path,
            "local_path": local_path,
            "active_path": existing_path or drive_path,
            "status": location_type or "needs_download",
        })

    # 3. Text Encoders & Clip Vision
    for enc_key_name, enc_type in [("clip", "text_encoder1"), ("clip2", "text_encoder2"), ("clip_vision", "clip_vision"), ("adapter", "adapter")]:
        reg_key = info.get(enc_key_name)
        if reg_key and reg_key in TEXT_ENCODER_REGISTRY:
            enc_url = TEXT_ENCODER_REGISTRY[reg_key]
            fname = enc_url.split("/")[-1].split("?")[0]
            drive_path = os.path.join(base_dir, "models", "text_encoders", fname)
            local_path = os.path.join(local_dir, fname)

            existing_path, location_type = find_existing_file_across_storage(
                [fname],
                [
                    os.path.join(base_dir, "models", "text_encoders"),
                    os.path.join(base_dir, "models"),
                    local_dir,
                ],
            )

            components.append({
                "name": f"Encoder ({reg_key})",
                "type": enc_type,
                "filename": fname,
                "url": enc_url,
                "drive_path": drive_path,
                "local_path": local_path,
                "active_path": existing_path or drive_path,
                "status": location_type or "needs_download",
            })

    return components
