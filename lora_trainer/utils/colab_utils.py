"""
Google Colab Utilities
Các tiện ích đặc thù cho môi trường Google Colab: Mount Drive, Auto Disconnect và Colab Port Proxy native.
"""

import os
import time
import socket
from typing import Optional


def mount_google_drive(mount_point: str = "/content/drive") -> bool:
    """Kết nối an toàn với Google Drive."""
    try:
        from google.colab import drive
        if not os.path.exists(mount_point):
            drive.mount(mount_point)
            print("✅ Đã kết nối thành công Google Drive!")
        else:
            print("✔️ Google Drive đã được mount sẵn.")
        return True
    except ImportError:
        print("ℹ️ Không ở trong môi trường Google Colab. Bỏ qua bước Mount Drive.")
        return False
    except Exception as e:
        print(f"⚠️ Lỗi khi mount Google Drive: {e}")
        return False


def auto_disconnect(delay_seconds: int = 180, enabled: bool = True) -> None:
    """Tự động ngắt kết nối runtime Colab sau khi hoàn thành để tiết kiệm Compute Units."""
    if not enabled:
        return

    print(f"\n⏳ Kích hoạt Auto-Disconnect! Hệ thống sẽ tự ngắt kết nối sau {delay_seconds} giây...")
    try:
        time.sleep(delay_seconds)
        from google.colab import runtime
        runtime.unassign()
        print("🔌 Đã ngắt kết nối Google Colab runtime thành công.")
    except Exception as e:
        print(f"ℹ️ Không thể gọi auto-disconnect: {e}")


def launch_colab_proxy(port: int = 8675, as_window: bool = True) -> Optional[str]:
    """
    Tạo đường dẫn kết nối trực tiếp đến cổng dịch vụ nội bộ (WebUI) trên Google Colab
    sử dụng cơ chế Colab Kernel Port Proxy gốc (không cần Cloudflare/Ngrok).
    """
    try:
        from google.colab.output import eval_js
        from IPython.display import display, HTML

        proxy_url = eval_js(f"google.colab.kernel.proxyPort({port})")
        
        html_ui = f"""
        <div style="background-color: #1e1e2f; padding: 16px; border-radius: 8px; border: 1px solid #4a4a6a; margin: 10px 0;">
            <h3 style="color: #4ade80; margin-top: 0;">🌐 WebUI đã sẵn sàng qua Google Colab Proxy!</h3>
            <p style="color: #e2e8f0; font-size: 14px;">Bấm vào liên kết bên dưới để mở giao diện WebUI trực tiếp trong tab mới:</p>
            <a href="{proxy_url}" target="_blank" style="display: inline-block; background-color: #3b82f6; color: white; padding: 10px 20px; font-weight: bold; text-decoration: none; border-radius: 6px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
                👉 Mở WebUI ({proxy_url})
            </a>
        </div>
        """
        display(HTML(html_ui))
        return proxy_url

    except ImportError:
        print(f"ℹ️ Local Web Server chạy tại: http://127.0.0.1:{port}")
        return f"http://127.0.0.1:{port}"
    except Exception as e:
        print(f"⚠️ Không thể tạo link Colab Proxy: {e}")
        return f"http://127.0.0.1:{port}"
