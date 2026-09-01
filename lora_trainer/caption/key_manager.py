"""
API Key Vault & Multi-Platform Manager
Quản lý, phân loại và tự động lưu trữ API Key theo từng nền tảng (Gemini, HuggingFace, WandB, OpenAI, Civitai).
Hỗ trợ tự động ghi nhớ sau lần nhập đầu tiên, chọn từ danh sách đã lưu và hiển thị Dashboard trực quan trên Google Colab.
"""

import os
import json
import datetime
from typing import Optional, Dict, List, Any


PRIMARY_VAULT_PATH = "/content/drive/MyDrive/TranningLorasData/config/api_vault.json"
FALLBACK_VAULT_PATHS = [
    PRIMARY_VAULT_PATH,
    "/content/drive/MyDrive/AI_Config/api_keys.json",
    "/content/drive/MyDrive/SD-Data/Setting/API_key_for_sdvn_comfy_node.json",
    os.path.expanduser("~/.config/tranning_loras/api_vault.json"),
]

# Danh mục nền tảng hỗ trợ
SUPPORTED_PLATFORMS = {
    "gemini": {
        "display_name": "Google Gemini / AI Studio",
        "env_var": "GEMINI_API_KEY",
        "doc_url": "https://aistudio.google.com",
    },
    "huggingface": {
        "display_name": "Hugging Face (HF_TOKEN)",
        "env_var": "HF_TOKEN",
        "doc_url": "https://huggingface.co/settings/tokens",
    },
    "wandb": {
        "display_name": "Weights & Biases (WandB)",
        "env_var": "WANDB_API_KEY",
        "doc_url": "https://wandb.ai/authorize",
    },
    "openai": {
        "display_name": "OpenAI (GPT-4o)",
        "env_var": "OPENAI_API_KEY",
        "doc_url": "https://platform.openai.com/api-keys",
    },
    "civitai": {
        "display_name": "Civitai (API Token)",
        "env_var": "CIVITAI_API_TOKEN",
        "doc_url": "https://civitai.com/user/account",
    },
}


def mask_key(key: Optional[str]) -> str:
    """Ẩn một phần key để hiển thị an toàn trên giao diện (VD: AIzaSy...4xK9)."""
    if not key or not isinstance(key, str):
        return ""
    key_str = key.strip()
    if len(key_str) <= 8:
        return "****"
    return f"{key_str[:6]}...{key_str[-4:]}"


def _get_active_vault_path() -> str:
    """Xác định đường dẫn lưu trữ vault khả dụng nhất (ưu tiên Google Drive)."""
    if os.path.exists("/content/drive/MyDrive"):
        os.makedirs(os.path.dirname(PRIMARY_VAULT_PATH), exist_ok=True)
        return PRIMARY_VAULT_PATH

    for p in FALLBACK_VAULT_PATHS:
        if os.path.exists(p):
            return p

    if FALLBACK_VAULT_PATHS:
        target = FALLBACK_VAULT_PATHS[0]
        os.makedirs(os.path.dirname(target), exist_ok=True)
        return target

    fallback_local = os.path.expanduser("~/.config/tranning_loras/api_vault.json")
    os.makedirs(os.path.dirname(fallback_local), exist_ok=True)
    return fallback_local


def load_vault() -> Dict[str, Any]:
    """Đọc toàn bộ dữ liệu Vault API Key từ Google Drive hoặc local."""
    for p in FALLBACK_VAULT_PATHS:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return _normalize_vault_structure(data)
            except Exception:
                continue

    return {plat: {"default": "", "items": []} for plat in SUPPORTED_PLATFORMS}


def _normalize_vault_structure(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Chuẩn hóa dữ liệu cũ (key-value phẳng) sang cấu trúc phân loại mới."""
    normalized = {plat: {"default": "", "items": []} for plat in SUPPORTED_PLATFORMS}

    for key, val in raw_data.items():
        k_lower = key.lower().strip()
        matched_plat = None
        for plat in SUPPORTED_PLATFORMS:
            if plat in k_lower or k_lower in plat:
                matched_plat = plat
                break

        if not matched_plat:
            matched_plat = k_lower
            if matched_plat not in normalized:
                normalized[matched_plat] = {"default": "", "items": []}

        if isinstance(val, dict):
            default_k = val.get("default", "")
            items = val.get("items", [])
            normalized[matched_plat] = {
                "default": default_k,
                "items": items if isinstance(items, list) else [],
            }
            if default_k and not items:
                normalized[matched_plat]["items"].append({
                    "label": "Mặc định",
                    "key": default_k,
                    "created_at": datetime.date.today().isoformat(),
                })
        elif isinstance(val, str) and val.strip():
            normalized[matched_plat]["default"] = val.strip()
            normalized[matched_plat]["items"] = [{
                "label": "Mặc định",
                "key": val.strip(),
                "created_at": datetime.date.today().isoformat(),
            }]

    return normalized


def save_vault(vault_data: Dict[str, Any]) -> bool:
    """Lưu toàn bộ Vault API Key vào tệp cấu hình trên Google Drive."""
    target_path = _get_active_vault_path()
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(vault_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[Cảnh báo Vault] Không thể lưu API Vault: {e}")
        return False


def save_api_key(
    service: str,
    api_key: str,
    label: Optional[str] = None,
    set_default: bool = True,
) -> bool:
    """Lưu 1 API Key mới vào Vault theo nền tảng, tự động gán nhãn và làm mặc định."""
    if not api_key or not api_key.strip():
        return False

    clean_key = api_key.strip()
    service_norm = service.lower().strip()

    vault = load_vault()
    if service_norm not in vault:
        vault[service_norm] = {"default": "", "items": []}

    plat_data = vault[service_norm]
    existing_items = plat_data.get("items", [])

    # Kiểm tra xem key đã có sẵn chưa
    found = False
    for item in existing_items:
        if item.get("key") == clean_key:
            found = True
            if label:
                item["label"] = label
            break

    if not found:
        item_label = label or f"Key #{len(existing_items) + 1} ({mask_key(clean_key)})"
        existing_items.append({
            "label": item_label,
            "key": clean_key,
            "created_at": datetime.date.today().isoformat(),
        })

    plat_data["items"] = existing_items
    if set_default or not plat_data.get("default"):
        plat_data["default"] = clean_key

    vault[service_norm] = plat_data
    saved = save_vault(vault)
    if saved:
        plat_name = SUPPORTED_PLATFORMS.get(service_norm, {}).get("display_name", service.upper())
        print(f"🔒 [Key Vault] Đã tự động lưu {plat_name}: {mask_key(clean_key)} vào Google Drive.")
    return saved


def get_saved_keys_list(service: str) -> List[Dict[str, Any]]:
    """Lấy danh sách toàn bộ API Key đã lưu của 1 nền tảng."""
    service_norm = service.lower().strip()
    vault = load_vault()
    plat_data = vault.get(service_norm, {"default": "", "items": []})
    default_key = plat_data.get("default", "")

    res = []
    for item in plat_data.get("items", []):
        k = item.get("key", "")
        res.append({
            "label": item.get("label", "Key"),
            "key": k,
            "masked": mask_key(k),
            "is_default": (k == default_key and bool(k)),
            "created_at": item.get("created_at", ""),
        })
    return res


def get_api_key(
    service: str,
    user_provided_key: Optional[str] = None,
    auto_save: bool = True,
    label: Optional[str] = None,
) -> Optional[str]:
    """
    Lấy API key theo thứ tự ưu tiên thông minh:
    1. Key được người dùng nhập trực tiếp (Nếu có -> Tự động lưu vào Drive cho các lần sau).
    2. Nếu người dùng chọn từ danh sách đã lưu (hoặc để trống) -> Lấy key mặc định đã lưu trong Vault.
    3. Biến môi trường hệ thống.
    4. Google Colab Userdata Secrets.
    """
    service_norm = service.lower().strip()
    vault = load_vault()
    plat_data = vault.get(service_norm, {"default": "", "items": []})

    # 1. Người dùng nhập key cụ thể hoặc chọn từ danh sách
    if user_provided_key and user_provided_key.strip():
        input_val = user_provided_key.strip()
        
        # Nếu người dùng chọn từ menu đã lưu
        for item in plat_data.get("items", []):
            if input_val in item.get("label", "") or input_val == item.get("masked"):
                return item.get("key")

        # Người dùng nhập key mới thực tế
        if auto_save:
            save_api_key(service_norm, input_val, label=label, set_default=True)
        return input_val

    # 2. Lấy key mặc định đã lưu trong Vault trên Google Drive
    saved_default = plat_data.get("default", "")
    if saved_default and saved_default.strip():
        return saved_default.strip()

    # 3. Biến môi trường
    env_info = SUPPORTED_PLATFORMS.get(service_norm, {})
    env_var = env_info.get("env_var", f"{service.upper()}_API_KEY")
    if env_var and os.environ.get(env_var):
        val = os.environ[env_var].strip()
        if auto_save and val:
            save_api_key(service_norm, val, label="From Env", set_default=True)
        return val

    # 4. Google Colab Userdata Secrets
    try:
        from google.colab import userdata
        val = userdata.get(service.upper()) or (env_var and userdata.get(env_var))
        if val and val.strip():
            val = val.strip()
            if auto_save:
                save_api_key(service_norm, val, label="Colab Secret", set_default=True)
            return val
    except Exception:
        pass

    return None


def display_key_vault_dashboard() -> None:
    """In bảng quản lý trực quan toàn bộ API Key đã lưu trên màn hình Colab."""
    vault = load_vault()
    print("\n" + "=" * 65)
    print("🔐 BẢNG ĐIỀU KHIỂN API KEY VAULT (ĐÃ LƯU TRÊN GOOGLE DRIVE)")
    print("=" * 65)

    has_any = False
    for plat, info in SUPPORTED_PLATFORMS.items():
        plat_data = vault.get(plat, {"default": "", "items": []})
        items = plat_data.get("items", [])
        default_key = plat_data.get("default", "")
        
        print(f"\n📌 {info['display_name']}:")
        if not items and not default_key:
            print("   (Chưa có key nào được lưu)")
        else:
            has_any = True
            for i, it in enumerate(items, 1):
                k = it.get("key", "")
                is_def = "⭐ [Mặc định]" if k == default_key else "  "
                print(f"   {is_def} #{i} {it.get('label')}: {mask_key(k)} (Ngày tạo: {it.get('created_at', 'N/A')})")

    print("\n" + "-" * 65)
    if has_any:
        print("💡 Mẹo: Ở các bước sau, bạn có thể để trống ô API Key để tự động dùng Key Mặc định!")
    else:
        print("💡 Mẹo: Khi nhập API Key lần đầu, hệ thống sẽ tự động lưu lại trên Google Drive.")
    print("=" * 65 + "\n")


def render_interactive_key_manager() -> None:
    """Tạo giao diện tương tác (Interactive Widgets) trên Colab để thêm/sửa/chọn key."""
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output

        vault = load_vault()
        out = widgets.Output()

        def refresh_ui():
            with out:
                clear_output(wait=True)
                display_key_vault_dashboard()

        service_dropdown = widgets.Dropdown(
            options=[(v["display_name"], k) for k, v in SUPPORTED_PLATFORMS.items()],
            description="Nền tảng:",
        )
        key_input = widgets.Password(description="API Key:")
        label_input = widgets.Text(description="Tên gợi nhớ:", placeholder="VD: Key Chính / Key Phụ")
        save_btn = widgets.Button(description="💾 Lưu vào Vault", button_style="success")

        def on_save_click(b):
            srv = service_dropdown.value
            k_val = key_input.value.strip()
            lbl = label_input.value.strip() or None
            if k_val:
                save_api_key(srv, k_val, label=lbl, set_default=True)
                key_input.value = ""
                label_input.value = ""
                refresh_ui()

        save_btn.on_click(on_save_click)
        box = widgets.VBox([
            widgets.HBox([service_dropdown, key_input]),
            widgets.HBox([label_input, save_btn]),
            out,
        ])
        display(box)
        refresh_ui()
    except Exception:
        display_key_vault_dashboard()

