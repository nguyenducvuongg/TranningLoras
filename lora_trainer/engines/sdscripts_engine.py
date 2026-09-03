"""
Kohya sd-scripts Runner Engine
Quản lý cài đặt, cập nhật kho mã nguồn kohya-ss/sd-scripts và thực thi huấn luyện
cho các dòng mô hình SDXL (Pony, Illustrious, Animagine), SD 1.5 và SD 3.5.
"""

import os
import sys
import subprocess
from typing import Optional
from ..core.base_engine import BaseTrainerEngine

SDSCRIPTS_REPO_URL = "https://github.com/kohya-ss/sd-scripts.git"
DEFAULT_SDSCRIPTS_DIR = "/content/sd-scripts"


def execute_command_stream(command_str: str, cwd: str) -> bool:
    """Thực thi câu lệnh terminal và truyền trực tiếp log ra console."""
    print(f"\n=======================================================")
    print(f"💻 ĐANG CHẠY: {command_str}")
    print(f"📂 Thư mục: {cwd}")
    print(f"=======================================================\n")

    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{cwd}:{existing_pp}".strip(":")
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        command_str,
        cwd=cwd,
        env=env,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            print(line, end="")
            sys.stdout.flush()

    process.wait()
    if process.returncode != 0:
        print(f"\n❌ Lỗi khi thực thi lệnh (Mã lỗi {process.returncode})!")
        return False
    return True


class SdScriptsEngine(BaseTrainerEngine):
    """Adapter thực thi Kohya sd-scripts."""

    def __init__(self, engine_dir: str = DEFAULT_SDSCRIPTS_DIR):
        super().__init__("sdscripts", engine_dir)

    def setup_repository(self) -> str:
        """Clone hoặc kéo commit mới nhất của sd-scripts."""
        if not os.path.exists(self.engine_dir):
            print(f"📦 Đang tải kho mã nguồn Kohya sd-scripts từ GitHub...")
            subprocess.run(["git", "clone", "--recurse-submodules", SDSCRIPTS_REPO_URL, self.engine_dir], check=True)
        else:
            print(f"🔄 Đang cập nhật sd-scripts lên bản mới nhất...")
            try:
                subprocess.run(["git", "-C", self.engine_dir, "pull"], check=False)
            except Exception as e:
                print(f"⚠️ Cảnh báo update sd-scripts: {e}")

        core_deps = [
            "voluptuous",
            "imagesize",
            "einops",
            "ftfy",
            "albumentations",
            "opencv-python-headless",
            "toml",
        ]
        missing = []
        for dep in core_deps:
            module_name = "cv2" if dep == "opencv-python-headless" else dep
            try:
                __import__(module_name)
            except ImportError:
                missing.append(dep)
        if missing:
            print(f"📦 Đang tự động bổ sung các gói bắt buộc cho sd-scripts: {', '.join(missing)}...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + missing, check=False)
            except Exception as e_dep:
                print(f"⚠️ Cảnh báo cài đặt dependency: {e_dep}")

        return self.engine_dir

    def is_installed(self) -> bool:
        return os.path.exists(os.path.join(self.engine_dir, "sdxl_train_network.py"))

    def run_training(self, train_command: str) -> bool:
        """Kích hoạt lệnh huấn luyện."""
        self.setup_repository()
        return execute_command_stream(train_command, self.engine_dir)


def run_sdscripts_pipeline(train_cmd: str, sdscripts_dir: str = DEFAULT_SDSCRIPTS_DIR) -> bool:
    """Hàm tiện ích chạy trực tiếp sd-scripts."""
    engine = SdScriptsEngine(sdscripts_dir)
    return engine.run_training(train_cmd)
