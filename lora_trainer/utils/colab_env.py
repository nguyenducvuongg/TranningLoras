"""
Google Colab Environment Utilities
Quản lý mount Google Drive, cài đặt các gói phụ thuộc hệ thống và tự động ngắt kết nối phiên Colab.
"""

import os
import sys
import subprocess
import time


def mount_google_drive(mount_point: str = "/content/drive") -> bool:
    """Mount Google Drive một cách an toàn."""
    try:
        from google.colab import drive
        if not os.path.exists(mount_point):
            drive.mount(mount_point)
            print("✅ Đã kết nối thành công Google Drive!")
        return True
    except Exception as e:
        print(f"⚠️ Không thể mount Google Drive ({e})")
        return False


def install_colab_prerequisites(quiet: bool = True) -> bool:
    """Cài đặt các gói phụ thuộc hệ thống cho Google Colab."""
    flags = "-qq" if quiet else ""
    try:
        subprocess.run(f"apt-get update {flags} && apt-get install -y {flags} aria2 ffmpeg", shell=True, check=False)
        return True
    except Exception as e:
        print(f"⚠️ Lỗi cài đặt apt packages: {e}")
        return False


def auto_disconnect(force: bool = False) -> None:
    """Tự động ngắt kết nối Runtime Colab để tiết kiệm Compute Units sau khi hoàn tất huấn luyện."""
    if not force:
        return
    print("🔌 Đang tự động ngắt kết nối phiên Google Colab...")
    try:
        from google.colab import runtime
        runtime.unassign()
    except Exception:
        pass
