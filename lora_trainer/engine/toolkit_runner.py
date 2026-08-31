"""
Ostris AI-Toolkit Runner
Quản lý quy trình cài đặt và kích hoạt tiến trình huấn luyện qua AI-Toolkit.
"""

import os
import sys
import subprocess
from typing import Optional

TOOLKIT_REPO_URL = "https://github.com/ostris/ai-toolkit.git"
DEFAULT_TOOLKIT_DIR = "/content/ai-toolkit"


def setup_toolkit_repo(toolkit_dir: str = DEFAULT_TOOLKIT_DIR) -> str:
    """Tự động clone hoặc cập nhật kho mã nguồn Ostris AI-Toolkit."""
    if not os.path.exists(toolkit_dir):
        print(f"📦 Đang tải kho mã nguồn AI-Toolkit từ GitHub...")
        subprocess.run(["git", "clone", "--recurse-submodules", TOOLKIT_REPO_URL, toolkit_dir], check=True)
    else:
        print(f"🔄 Đang cập nhật AI-Toolkit...")
        try:
            subprocess.run(["git", "-C", toolkit_dir, "pull"], check=False)
        except Exception:
            pass

    return toolkit_dir


def run_toolkit_pipeline(config_yaml_path: str, toolkit_dir: str = DEFAULT_TOOLKIT_DIR) -> bool:
    """Khởi chạy huấn luyện qua AI-Toolkit với tệp YAML cấu hình."""
    if not os.path.exists(config_yaml_path):
        raise FileNotFoundError(f"Không tìm thấy file cấu hình YAML: {config_yaml_path}")

    setup_toolkit_repo(toolkit_dir)

    run_cmd = f"python run.py {config_yaml_path}"
    print(f"\n=======================================================")
    print(f"💻 ĐANG CHẠY AI-TOOLKIT: {run_cmd}")
    print(f"📂 Thư mục: {toolkit_dir}")
    print(f"=======================================================\n")

    process = subprocess.Popen(
        run_cmd,
        cwd=toolkit_dir,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    for line in process.stdout:
        print(line, end="")
        sys.stdout.flush()

    process.wait()
    if process.returncode != 0:
        print(f"\n❌ Lỗi khi huấn luyện AI-Toolkit (Mã lỗi {process.returncode})!")
        return False

    print("\n🎉 HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG!")
    return True
