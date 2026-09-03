"""
Kohya Musubi-Tuner Runner Engine
Quản lý quy trình cài đặt, tiền cache (Latents, Text Encoders) và kích hoạt tiến trình huấn luyện qua Accelerate
cho Wan 2.1/2.2, FLUX.2 Klein, Qwen-Image, Z-Image, Krea2.
"""

import os
import sys
import subprocess
from typing import Optional
from ..core.base_engine import BaseTrainerEngine

MUSUBI_REPO_URL = "https://github.com/kohya-ss/musubi-tuner.git"
DEFAULT_MUSUBI_DIR = "/content/musubi-tuner"


def execute_command_stream(command_str: str, cwd: str) -> bool:
    """Thực thi câu lệnh terminal và truyền trực tiếp log ra console."""
    print(f"\n=======================================================")
    print(f"💻 ĐANG CHẠY: {command_str}")
    print(f"📂 Thư mục: {cwd}")
    print(f"=======================================================\n")

    env = os.environ.copy()
    src_dir = os.path.join(cwd, "src")
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}:{cwd}:{existing_pp}".strip(":")
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


class MusubiEngine(BaseTrainerEngine):
    """Adapter thực thi Kohya Musubi-Tuner."""

    def __init__(self, engine_dir: str = DEFAULT_MUSUBI_DIR):
        super().__init__("musubi", engine_dir)

    def setup_repository(self) -> str:
        """Clone hoặc kéo commit mới nhất của musubi-tuner."""
        if not os.path.exists(self.engine_dir):
            print(f"📦 Đang tải kho mã nguồn Musubi-Tuner mới nhất từ GitHub...")
            subprocess.run(["git", "clone", "--recurse-submodules", MUSUBI_REPO_URL, self.engine_dir], check=True)
        else:
            print(f"🔄 Đang cập nhật Musubi-Tuner lên bản mới nhất (git pull & submodules)...")
            try:
                subprocess.run(["git", "-C", self.engine_dir, "checkout", "main"], check=False)
                subprocess.run(["git", "-C", self.engine_dir, "pull"], check=False)
                subprocess.run(["git", "-C", self.engine_dir, "submodule", "update", "--init", "--recursive"], check=False)
            except Exception as e:
                print(f"⚠️ Cảnh báo khi update: {e}")

        # 1. Đảm bảo các gói phụ thuộc bắt buộc mà musubi-tuner import luôn sẵn sàng (chống lỗi No module named 'av')
        core_deps = [
            "av",
            "easydict",
            "voluptuous",
            "einops",
            "ftfy",
            "sentencepiece",
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
            print(f"📦 Đang tự động bổ sung các gói bắt buộc cho Musubi-Tuner: {', '.join(missing)}...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + missing, check=False)
            except Exception as e_dep:
                print(f"⚠️ Cảnh báo cài đặt dependency: {e_dep}")

        # 2. Cài đặt musubi-tuner ở chế độ editable với --no-deps để tránh hạ cấp / xung đột torch
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", self.engine_dir], check=False)
        except Exception:
            pass

        return self.engine_dir

    def is_installed(self) -> bool:
        return os.path.exists(os.path.join(self.engine_dir, "wan_train_network.py")) or os.path.exists(os.path.join(self.engine_dir, "src"))

    def run_training(
        self,
        cache_latents_cmd: Optional[str] = None,
        cache_text_encoder_cmd: Optional[str] = None,
        train_cmd: Optional[str] = None,
        skip_cache: bool = False,
    ) -> bool:
        """
        Chạy trọn vẹn 3 giai đoạn của Musubi-Tuner:
        1. Pre-cache VAE Latents
        2. Pre-cache Text Encoders
        3. Accelerate Training
        """
        self.setup_repository()

        if cache_latents_cmd and not skip_cache:
            print("\n🔹 [GIAI ĐOẠN 1/3]: PRE-CACHE VAE LATENTS...")
            ok = execute_command_stream(cache_latents_cmd, self.engine_dir)
            if not ok:
                return False

        if cache_text_encoder_cmd and not skip_cache:
            print("\n🔹 [GIAI ĐOẠN 2/3]: PRE-CACHE TEXT ENCODERS...")
            ok = execute_command_stream(cache_text_encoder_cmd, self.engine_dir)
            if not ok:
                return False

        if train_cmd:
            print("\n🔹 [GIAI ĐOẠN 3/3]: HUẤN LUYỆN ACCELERATE...")
            ok = execute_command_stream(train_cmd, self.engine_dir)
            if not ok:
                return False

        print("\n🎉 HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG!")
        return True


def run_musubi_pipeline(
    musubi_dir: str = DEFAULT_MUSUBI_DIR,
    cache_latents_cmd: Optional[str] = None,
    cache_text_encoder_cmd: Optional[str] = None,
    train_cmd: Optional[str] = None,
    skip_cache: bool = False,
) -> bool:
    """Hàm tiện ích chạy trực tiếp Musubi-Tuner pipeline."""
    engine = MusubiEngine(musubi_dir)
    return engine.run_training(
        cache_latents_cmd=cache_latents_cmd,
        cache_text_encoder_cmd=cache_text_encoder_cmd,
        train_cmd=train_cmd,
        skip_cache=skip_cache,
    )
