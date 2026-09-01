"""
Abstract Base Trainer Engine
Giao diện trừu tượng (Interface) định nghĩa các phương thức chuẩn mực mà mọi Engine huấn luyện (Kohya sd-scripts, Musubi, AI-Toolkit) phải tuân theo.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseTrainerEngine(ABC):
    """Lớp cơ sở cho các Adapter Engine huấn luyện."""

    def __init__(self, engine_name: str, engine_dir: str):
        self.engine_name = engine_name
        self.engine_dir = engine_dir

    @abstractmethod
    def setup_repository(self) -> str:
        """Clone hoặc kéo commit mới nhất của repo engine."""
        pass

    @abstractmethod
    def is_installed(self) -> bool:
        """Kiểm tra engine đã cài đặt sẵn sàng chưa."""
        pass

    @abstractmethod
    def run_training(self, **kwargs) -> bool:
        """Thực thi toàn bộ tiến trình huấn luyện."""
        pass
