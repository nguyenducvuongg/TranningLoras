"""
Google Colab Utilities
Các tiện ích đặc thù cho môi trường Google Colab: Mount Drive, Auto Disconnect và Cloudflare Tunnel.
"""

import os
import time
import socket
import threading
import subprocess
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


def _tunnel_worker(port: int):
    """Worker luồng nền cho Cloudflare tunnel."""
    # Đợi port mở
    while True:
        time.sleep(0.5)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        res = sock.connect_ex(("127.0.0.1", port))
        sock.close()
        if res == 0:
            break

    cmd = ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    for line in process.stderr:
        decoded = line.decode()
        if "trycloudflare.com" in decoded:
            url_start = decoded.find("https://")
            if url_start != -1:
                url = decoded[url_start:].strip().split()[0]
                print(f"\n\033[92m🔗 Link WebUI Cloudflare Online: {url}\033[0m\n")


def launch_cloudflare_tunnel(port: int = 8675) -> None:
    """Khởi chạy Cloudflare Tunnel cho WebUI."""
    t = threading.Thread(target=_tunnel_worker, args=(port,), daemon=True)
    t.start()
