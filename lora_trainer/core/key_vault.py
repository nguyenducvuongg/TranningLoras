"""
API Key Vault Manager
Quản lý bảo mật các khóa API (Gemini, HuggingFace, WandB, CivitAI, OpenAI)
Lưu trữ trên Google Drive (/content/drive/MyDrive/TranningLorasData/config/api_vault.json)
để tự động tái sử dụng trong các phiên huấn luyện kế tiếp mà không cần nhập lại.
"""

import os
import json
from typing import Dict, Any, Optional

DEFAULT_VAULT_PATH = "/content/drive/MyDrive/TranningLorasData/config/api_vault.json"


def get_vault_path() -> str:
    """Trả về đường dẫn tệp api_vault.json."""
    return DEFAULT_VAULT_PATH


def load_api_vault(vault_path: Optional[str] = None) -> Dict[str, Any]:
    """Tải toàn bộ dữ liệu từ tệp API Vault."""
    path = vault_path or get_vault_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception as e:
        print(f"⚠️ Không thể đọc API Vault ({e})")
    return {}


def save_api_key(
    platform: str,
    key_value: str,
    label: Optional[str] = None,
    vault_path: Optional[str] = None,
    set_default: bool = True,
) -> bool:
    """
    Lưu khóa API mới vào Vault.
    Hỗ trợ lưu nhiều key với nhãn (label) và tự động đặt làm default key.
    """
    if not key_value or not key_value.strip():
        return False

    platform = platform.lower().strip()
    key_value = key_value.strip()
    path = vault_path or get_vault_path()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    vault = load_api_vault(path)

    if platform not in vault:
        vault[platform] = {
            "default": key_value,
            "keys": {}
        }
    elif not isinstance(vault[platform], dict):
        vault[platform] = {
            "default": str(vault[platform]),
            "keys": {}
        }

    key_label = label.strip() if label and label.strip() else f"key_{len(vault[platform].get('keys', {})) + 1}"
    if "keys" not in vault[platform]:
        vault[platform]["keys"] = {}

    vault[platform]["keys"][key_label] = key_value

    if set_default or "default" not in vault[platform] or not vault[platform]["default"]:
        vault[platform]["default"] = key_value

    # Đồng bộ với biến môi trường hệ thống
    env_map = {
        "gemini": "GEMINI_API_KEY",
        "huggingface": "HF_TOKEN",
        "wandb": "WANDB_API_KEY",
        "openai": "OPENAI_API_KEY",
        "civitai": "CIVITAI_API_KEY",
    }
    if platform in env_map:
        os.environ[env_map[platform]] = key_value

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(vault, f, indent=2, ensure_ascii=False)
        print(f"🔐 Đã lưu an toàn API Key cho [{platform.upper()}] vào Google Drive!")
        return True
    except Exception as e:
        print(f"❌ Lỗi khi lưu API Vault: {e}")
        return False


def get_api_key(
    platform: str,
    key_label: Optional[str] = None,
    vault_path: Optional[str] = None,
) -> Optional[str]:
    """
    Lấy API key theo thứ tự ưu tiên:
    1. Biến môi trường hệ thống (nếu đã đặt)
    2. Tệp API Vault trên Google Drive
    """
    platform = platform.lower().strip()

    env_map = {
        "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "huggingface": ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"],
        "wandb": ["WANDB_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
        "civitai": ["CIVITAI_API_KEY", "CIVITAI_TOKEN"],
    }

    if platform in env_map:
        for env_var in env_map[platform]:
            val = os.environ.get(env_var)
            if val and val.strip():
                return val.strip()

    vault = load_api_vault(vault_path)
    if platform not in vault:
        return None

    plat_data = vault[platform]
    if isinstance(plat_data, str):
        return plat_data.strip()
    if isinstance(plat_data, dict):
        if key_label and key_label in plat_data.get("keys", {}):
            return plat_data["keys"][key_label].strip()
        return plat_data.get("default", None)

    return None


def mask_key(k: Optional[str]) -> str:
    """Ẩn bớt ký tự của key để hiển thị an toàn trên dashboard."""
    if not k:
        return "None"
    k = k.strip()
    if len(k) <= 8:
        return "****"
    return f"{k[:4]}...{k[-4:]}"


def display_key_vault_dashboard(vault_path: Optional[str] = None) -> None:
    """Hiển thị bảng tổng hợp các API Key đang có trong Vault."""
    vault = load_api_vault(vault_path)
    print("=" * 60)
    print("🔐 BẢNG ĐIỀU KHIỂN KHÓA BẢO MẬT (API KEY VAULT)")
    print("=" * 60)
    platforms = ["gemini", "huggingface", "wandb", "openai", "civitai"]
    for plat in platforms:
        key = get_api_key(plat, vault_path=vault_path)
        status = f"✅ Đã cấu hình ({mask_key(key)})" if key else "⚪ Chưa cấu hình (Tùy chọn)"
        print(f" • {plat.upper():<12}: {status}")
    print("=" * 60 + "\n")
