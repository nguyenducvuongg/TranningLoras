"""
Universal Model Suite Resolver & Fetcher
Tự động quét kho lưu trữ vĩnh viễn trên Google Drive, phân giải toàn bộ các thành phần
cần thiết (Base Model/DiT/UNet, VAE, Text Encoders, Clip Vision) và tải siêu tốc các file còn thiếu.
"""

import os
from typing import Dict, Any, List, Optional
from .drive_manager import (
    DEFAULT_DRIVE_ROOT,
    LOCAL_CACHE_DIR,
    setup_storage_structure,
    get_model_component_paths,
    is_file_complete,
)
from .downloader import aria2_download, get_hf_token, get_civitai_key
from ..core.model_registry import get_model_info, VAE_REGISTRY, TEXT_ENCODER_REGISTRY


def download_model_suite(
    model_name: str,
    weights_dir: str = LOCAL_CACHE_DIR,
    hf_token: Optional[str] = None,
    civitai_key: Optional[str] = None,
    base_drive_dir: str = DEFAULT_DRIVE_ROOT,
    custom_url: Optional[str] = None,
) -> Dict[str, str]:
    """
    Tải toàn diện bộ model theo cấu hình:
    1. Kiểm tra Google Drive -> Nếu đã có đầy đủ: BỎ QUA 100% (0 giây chờ).
    2. Nếu thiếu: Tải thẳng vào thư mục tương ứng trên Google Drive bằng aria2 siêu tốc.
    3. Trả về Dictionary các đường dẫn tệp thực tế đã sẵn sàng để huấn luyện.
    """
    setup_storage_structure(base_drive_dir)
    components = get_model_component_paths(
        model_name=model_name,
        base_dir=base_drive_dir,
        local_dir=weights_dir,
        custom_url=custom_url,
    )

    token = hf_token if hf_token else get_hf_token()
    civ_key = civitai_key if civitai_key else get_civitai_key()
    resolved_paths: Dict[str, str] = {}

    print(f"\n=======================================================")
    print(f"📦 KIỂM TRA & PHÂN GIẢI TRỌNG SỐ CHO: {model_name}")
    print(f"=======================================================")

    for comp in components:
        comp_type = comp["type"]
        fname = comp["filename"]
        target_drive_dir = os.path.dirname(comp["drive_path"])
        os.makedirs(target_drive_dir, exist_ok=True)

        if comp["status"] in ["ready_drive", "ready_local"]:
            active = comp["active_path"]
            print(f"✔️ [ĐÃ CÓ SẴN] {comp['name']}: {active}")
            resolved_paths[comp_type] = active
            continue

        # Tiến hành tải xuống
        print(f"🚀 [CẦN TẢI] {comp['name']} -> {target_drive_dir}/{fname}")
        downloaded = None
        try:
            downloaded = aria2_download(
                url=comp["url"],
                destination_dir=target_drive_dir,
                filename=fname,
                token=token,
                civitai_key=civ_key,
            )
        except Exception as e:
            if comp.get("fallback_url"):
                print(f"⚠️ Thử lại với URL Fallback: {comp['fallback_url']}")
                try:
                    downloaded = aria2_download(
                        url=comp["fallback_url"],
                        destination_dir=target_drive_dir,
                        filename=fname,
                        token=token,
                        civitai_key=civ_key,
                    )
                except Exception as e_fb:
                    print(f"❌ Lỗi tải fallback: {e_fb}")
            else:
                print(f"❌ Lỗi khi tải {fname}: {e}")

        if downloaded and is_file_complete(downloaded):
            resolved_paths[comp_type] = downloaded
        elif is_file_complete(comp["drive_path"]):
            resolved_paths[comp_type] = comp["drive_path"]
        elif is_file_complete(comp.get("local_path")):
            resolved_paths[comp_type] = comp["local_path"]
        else:
            resolved_paths[comp_type] = comp["drive_path"]
            if comp_type == "dit":
                raise RuntimeError(
                    f"❌ Không thể tải thành công mô hình cốt lõi '{comp['name']}' ({fname})!\n"
                    f"Vui lòng kiểm tra lại kết nối mạng hoặc cung cấp HuggingFace Token / CivitAI API Key hợp lệ trong ô cấu hình."
                )

    print("=======================================================\n")
    return resolved_paths
