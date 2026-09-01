"""
Kohya Musubi-Tuner Runner
Quản lý quy trình cài đặt, tiền cache (Latents, Text Encoders) và kích hoạt tiến trình huấn luyện qua Accelerate.
Luôn đảm bảo cập nhật phiên bản mới nhất từ kho mã nguồn chính thức.
"""

import os
import sys
import subprocess
from typing import Dict, Any, Optional

MUSUBI_REPO_URL = "https://github.com/kohya-ss/musubi-tuner.git"
DEFAULT_MUSUBI_DIR = "/content/musubi-tuner"


def setup_musubi_repo(musubi_dir: str = DEFAULT_MUSUBI_DIR) -> str:
    """Tự động clone hoặc kéo commit mới nhất kèm submodule của Kohya Musubi-Tuner."""
    if not os.path.exists(musubi_dir):
        print(f"📦 Đang tải kho mã nguồn Musubi-Tuner mới nhất từ GitHub...")
        subprocess.run(["git", "clone", "--recurse-submodules", MUSUBI_REPO_URL, musubi_dir], check=True)
    else:
        print(f"🔄 Đang cập nhật Musubi-Tuner lên bản mới nhất (git pull & submodules)...")
        try:
            subprocess.run(["git", "-C", musubi_dir, "checkout", "main"], check=False)
            subprocess.run(["git", "-C", musubi_dir, "pull"], check=False)
            subprocess.run(["git", "-C", musubi_dir, "submodule", "update", "--init", "--recursive"], check=False)
        except Exception as e:
            print(f"⚠️ Cảnh báo khi update: {e}")

    # Đảm bảo cài đặt các phụ thuộc và cài đặt gói musubi-tuner ở chế độ editable
    req_file = os.path.join(musubi_dir, "requirements.txt")
    if os.path.exists(req_file):
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", req_file], check=False)
        except Exception:
            pass

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "voluptuous",
                "imagesize",
                "einops",
                "ftfy",
                "regex",
                "sentencepiece",
                "protobuf",
                "scipy",
                "wandb",
                "lion-pytorch",
                "prodigyopt",
                "albumentations",
            ],
            check=False,
        )
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", musubi_dir], check=False)
    except Exception:
        pass

    return musubi_dir


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

    process = subprocess.Popen(
        command_str,
        cwd=cwd,
        env=env,
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
        print(f"\n❌ Lỗi khi thực thi lệnh (Mã lỗi {process.returncode})!")
        return False
    return True


def run_musubi_pipeline(
    musubi_dir: str = DEFAULT_MUSUBI_DIR,
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
    setup_musubi_repo(musubi_dir)

    # Giai đoạn 1: Cache Latents
    if cache_latents_cmd and not skip_cache:
        print("\n🔹 [GIAI ĐOẠN 1/3]: PRE-CACHE VAE LATENTS...")
        ok = execute_command_stream(cache_latents_cmd, musubi_dir)
        if not ok:
            return False

    # Giai đoạn 2: Cache Text Encoders
    if cache_text_encoder_cmd and not skip_cache:
        print("\n🔹 [GIAI ĐOẠN 2/3]: PRE-CACHE TEXT ENCODERS...")
        ok = execute_command_stream(cache_text_encoder_cmd, musubi_dir)
        if not ok:
            return False

    # Giai đoạn 3: Huấn luyện chính
    if train_cmd:
        print("\n🔹 [GIAI ĐOẠN 3/3]: HUẤN LUYỆN LORA (ACCELERATE LAUNCH)...")
        ok = execute_command_stream(train_cmd, musubi_dir)
        if not ok:
            return False

    print("\n🎉 HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG!")
    return True
