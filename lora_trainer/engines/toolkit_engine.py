"""
Ostris AI-Toolkit Runner Engine
Quản lý cài đặt, cập nhật kho mã nguồn ostris/ai-toolkit và thực thi huấn luyện qua file YAML cấu hình.
"""

import os
import sys
import subprocess
from typing import Optional
from ..core.base_engine import BaseTrainerEngine
from ..ui.dashboard import get_dashboard

DEFAULT_TOOLKIT_DIR = "/content/ai-toolkit"
TOOLKIT_REPO_URL = "https://github.com/ostris/ai-toolkit.git"


def execute_command_stream(command_str: str, cwd: str, dashboard=None) -> bool:
    """Thực thi câu lệnh terminal và truyền trực tiếp log ra console & dashboard."""
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

    dash = dashboard or get_dashboard()

    if process.stdout:
        for line in iter(process.stdout.readline, ""):
            print(line, end="")
            sys.stdout.flush()
            if dash:
                dash.update_line(line)

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

    def run_training(self, config_yaml_path: str, dashboard=None) -> bool:
        """Kích hoạt tiến trình huấn luyện với YAML config."""
        if not os.path.exists(config_yaml_path):
            raise FileNotFoundError(f"Không tìm thấy file cấu hình YAML: {config_yaml_path}")

        dash = dashboard or get_dashboard()
        self.setup_repository()
        if dash:
            dash.skip_stage(1, reason="AI-Toolkit On-the-fly VAE")
            dash.skip_stage(2, reason="AI-Toolkit On-the-fly TE")
            dash.set_stage(3, "running", "Đang huấn luyện qua AI-Toolkit...")

        run_cmd = f"python run.py '{config_yaml_path}'"
        ok = execute_command_stream(run_cmd, self.engine_dir, dashboard=dash)
        if ok:
            if dash:
                dash.finish(success=True)
            print("\n🎉 HUẤN LUYỆN AI-TOOLKIT HOÀN TẤT THÀNH CÔNG!")
        else:
            if dash:
                dash.finish(success=False, message="AI-Toolkit gặp lỗi dừng")
        return ok


def run_toolkit_pipeline(config_yaml_path: str, toolkit_dir: str = DEFAULT_TOOLKIT_DIR, dashboard=None) -> bool:
    """Hàm tiện ích chạy trực tiếp AI-Toolkit."""
    engine = ToolkitEngine(toolkit_dir)
    return engine.run_training(config_yaml_path, dashboard=dashboard)
