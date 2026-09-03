"""
Ostris AI-Toolkit Runner Engine
Quản lý cài đặt, cập nhật kho mã nguồn ostris/ai-toolkit và thực thi huấn luyện qua file YAML cấu hình.
"""

import os
import sys
import subprocess
from typing import Optional
from ..core.base_engine import BaseTrainerEngine

TOOLKIT_REPO_URL = "https://github.com/ostris/ai-toolkit.git"
DEFAULT_TOOLKIT_DIR = "/content/ai-toolkit"


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
        print(f"\n❌ Lỗi khi thực thi AI-Toolkit (Mã lỗi {process.returncode})!")
        return False
    return True


class ToolkitEngine(BaseTrainerEngine):
    """Adapter thực thi Ostris AI-Toolkit."""

    def __init__(self, engine_dir: str = DEFAULT_TOOLKIT_DIR):
        super().__init__("toolkit", engine_dir)

    def setup_repository(self) -> str:
        """Clone hoặc kéo commit mới nhất của ai-toolkit."""
        if not os.path.exists(self.engine_dir):
            print(f"📦 Đang tải kho mã nguồn AI-Toolkit mới nhất từ GitHub...")
            subprocess.run(["git", "clone", "--recurse-submodules", TOOLKIT_REPO_URL, self.engine_dir], check=True)
        else:
            print(f"🔄 Đang cập nhật AI-Toolkit lên bản mới nhất...")
            try:
                subprocess.run(["git", "-C", self.engine_dir, "pull"], check=False)
                subprocess.run(["git", "-C", self.engine_dir, "submodule", "update", "--init", "--recursive"], check=False)
            except Exception as e:
                print(f"⚠️ Cảnh báo update AI-Toolkit: {e}")

        core_deps = ["yaml", "safetensors", "diffusers", "transformers"]
        missing = []
        for dep in core_deps:
            module_name = dep
            try:
                __import__(module_name)
            except ImportError:
                missing.append("pyyaml" if dep == "yaml" else dep)
        if missing:
            print(f"📦 Đang tự động bổ sung các gói bắt buộc cho AI-Toolkit: {', '.join(missing)}...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + missing, check=False)
            except Exception as e_dep:
                print(f"⚠️ Cảnh báo cài đặt dependency: {e_dep}")

        return self.engine_dir

    def is_installed(self) -> bool:
        return os.path.exists(os.path.join(self.engine_dir, "run.py"))

    def run_training(self, config_yaml_path: str) -> bool:
        """Kích hoạt tiến trình huấn luyện với YAML config."""
        if not os.path.exists(config_yaml_path):
            raise FileNotFoundError(f"Không tìm thấy file cấu hình YAML: {config_yaml_path}")

        self.setup_repository()
        run_cmd = f"python run.py '{config_yaml_path}'"
        ok = execute_command_stream(run_cmd, self.engine_dir)
        if ok:
            print("\n🎉 HUẤN LUYỆN AI-TOOLKIT HOÀN TẤT THÀNH CÔNG!")
        return ok


def run_toolkit_pipeline(config_yaml_path: str, toolkit_dir: str = DEFAULT_TOOLKIT_DIR) -> bool:
    """Hàm tiện ích chạy trực tiếp AI-Toolkit."""
    engine = ToolkitEngine(toolkit_dir)
    return engine.run_training(config_yaml_path)
