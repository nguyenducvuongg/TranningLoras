"""
API Key Manager
Quản lý API Key an toàn từ biến môi trường, Google Colab Secrets (userdata) hoặc tệp cấu hình trên Google Drive.
"""

import os
import json
from typing import Optional, Dict

DEFAULT_DRIVE_KEY_PATHS = [
    "/content/drive/MyDrive/AI_Config/api_keys.json",
    "/content/drive/MyDrive/SD-Data/Setting/API_key_for_sdvn_comfy_node.json",
    os.path.expanduser("~/.config/ai_keys.json"),
]


def get_api_key(service: str, user_provided_key: Optional[str] = None) -> Optional[str]:
    """
    Lấy API key theo thứ tự ưu tiên:
    1. Key được người dùng nhập trực tiếp trong Form.
    2. Biến môi trường (GEMINI_API_KEY, OPENAI_API_KEY, WANDB_API_KEY).
    3. Google Colab Secrets (google.colab.userdata.get).
    4. Tệp JSON cấu hình trên Google Drive.
    """
    service_norm = service.lower().strip()
    
    # 1. User provided key
    if user_provided_key and user_provided_key.strip():
        return user_provided_key.strip()

    # 2. Environment Variables
    env_keys = {
        "gemini": "GEMINI_API_KEY",
        "google": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
        "wandb": "WANDB_API_KEY",
        "huggingface": "HF_TOKEN",
        "civitai": "CIVITAI_API_TOKEN",
    }
    env_var = env_keys.get(service_norm)
    if env_var and os.environ.get(env_var):
        return os.environ[env_var].strip()

    # 3. Google Colab Userdata Secrets
    try:
        from google.colab import userdata
        val = userdata.get(service.upper()) or (env_var and userdata.get(env_var))
        if val:
            return val.strip()
    except Exception:
        pass

    # 4. JSON Files
    for p in DEFAULT_DRIVE_KEY_PATHS:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Match key
                for k, v in data.items():
                    if k.lower() in service_norm or service_norm in k.lower():
                        if isinstance(v, str) and v.strip():
                            return v.strip()
            except Exception:
                pass

    return None


def save_api_key(service: str, api_key: str, file_path: str = DEFAULT_DRIVE_KEY_PATHS[0]) -> bool:
    """Lưu API key vào tệp cấu hình để tái sử dụng trong các phiên làm việc sau."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        data[service] = api_key.strip()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"[Lỗi] Không thể lưu API key: {e}")
        return False
