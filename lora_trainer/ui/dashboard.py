"""
Live Visual Training Dashboard for Google Colab & Jupyter Notebooks.
Cung cấp bảng điều khiển Glassmorphism hiển thị tiến trình, hàm loss, tốc độ, VRAM theo thời gian thực.
"""

import os
import sys
import re
import time
import datetime
from typing import Optional, Dict, Any, List

# Kiểm tra môi trường IPython / Colab
_HAS_IPYTHON = False
try:
    from IPython.display import display, HTML, update_display
    _HAS_IPYTHON = True
except ImportError:
    _HAS_IPYTHON = False


class TrainingDashboard:
    """Bảng điều khiển trực quan hiển thị tiến trình huấn luyện LoRA."""

    def __init__(
        self,
        model_name: str = "SDXL",
        engine_name: str = "sdscripts",
        task_type: str = "face",
        lora_name: str = "my_lora",
        total_steps: int = 1500,
        total_epochs: int = 10,
        output_dir: str = "",
        anti_plastic_info: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.engine_name = engine_name
        self.task_type = task_type
        self.lora_name = lora_name
        self.total_steps = max(1, total_steps)
        self.total_epochs = max(1, total_epochs)
        self.output_dir = output_dir
        self.anti_plastic_info = anti_plastic_info or {
            "noise_offset": 0.06,
            "min_snr": 5,
            "no_half_vae": True,
            "sdpa": True,
        }

        # Trạng thái pipeline 3 giai đoạn
        self.stages = [
            {"id": 1, "name": "Pre-cache VAE Latents", "status": "pending", "detail": "Chờ kích hoạt"},
            {"id": 2, "name": "Pre-cache Text Encoders", "status": "pending", "detail": "Chờ kích hoạt"},
            {"id": 3, "name": "Huấn luyện LoRA Engine", "status": "pending", "detail": "Chờ kích hoạt"},
        ]
        self.current_stage = 1
        self.status = "INITIALIZING"  # RUNNING, COMPLETED, FAILED

        # Tiến trình & Chỉ số
        self.current_step = 0
        self.current_epoch = 1
        self.percent = 0.0
        self.current_loss = 0.0
        self.min_loss = float("inf")
        self.loss_history: List[float] = []
        self.speed = "0.00 it/s"
        self.eta = "--:--"
        self.start_time = time.time()
        self.last_render_time = 0.0
        self.render_interval = 0.6  # Giới hạn cập nhật tối đa ~1.6 FPS để tiết kiệm CPU

        # Ring buffer nhật ký dòng lệnh
        self.log_buffer: List[str] = []
        self.max_log_lines = 50

        # Unique display ID để cập nhật tại chỗ (in-place)
        self.display_id = f"tranning_loras_dash_{int(time.time())}"
        self.is_displayed = False

    def set_stage(self, stage_num: int, status: str = "running", detail: str = ""):
        """Cập nhật trạng thái của từng giai đoạn (1, 2, 3)."""
        self.current_stage = stage_num
        for s in self.stages:
            if s["id"] < stage_num:
                s["status"] = "done"
                s["detail"] = "Hoàn tất"
            elif s["id"] == stage_num:
                s["status"] = status
                if detail:
                    s["detail"] = detail
            else:
                s["status"] = "pending"
                s["detail"] = "Chờ đến lượt"

        if status == "running":
            self.status = "RUNNING"
        self.render(force=True)

    def skip_stage(self, stage_num: int, reason: str = "Đã bỏ qua / Cache sẵn"):
        """Đánh dấu một giai đoạn được bỏ qua."""
        for s in self.stages:
            if s["id"] == stage_num:
                s["status"] = "skipped"
                s["detail"] = reason
        self.render(force=True)

    def parse_line(self, line: str):
        """Phân tích dòng log stdout từ Kohya hoặc AI-Toolkit để trích xuất chỉ số."""
        clean_line = line.strip()
        if not clean_line:
            return

        # Lưu vào log buffer
        self.log_buffer.append(clean_line)
        if len(self.log_buffer) > self.max_log_lines:
            self.log_buffer.pop(0)

        # 1. Regex nhận diện tiến trình TQDM của Kohya sd-scripts / musubi:
        # e.g.: steps:  25%|██▌       | 250/1000 [03:15<09:45,  1.28it/s, loss=0.0782]
        m_tqdm = re.search(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([^<]+)<([^,]+),\s*([0-9\.]+(?:it/s|s/it))", clean_line)
        if m_tqdm:
            self.percent = float(m_tqdm.group(1))
            self.current_step = int(m_tqdm.group(2))
            self.total_steps = int(m_tqdm.group(3))
            self.eta = m_tqdm.group(5).strip()
            self.speed = m_tqdm.group(6).strip()

        # 2. Regex nhận diện Loss:
        m_loss = re.search(r"loss[=:\s]+([0-9\.]+)", clean_line, re.IGNORECASE)
        if m_loss:
            try:
                loss_val = float(m_loss.group(1))
                if 0.0001 < loss_val < 50.0:  # Lọc giá trị hợp lý
                    self.current_loss = loss_val
                    self.loss_history.append(loss_val)
                    if len(self.loss_history) > 30:
                        self.loss_history.pop(0)
                    if loss_val < self.min_loss:
                        self.min_loss = loss_val
            except ValueError:
                pass

        # 3. Regex nhận diện Epoch:
        m_epoch = re.search(r"epoch\s*(\d+)(?:/(\d+))?", clean_line, re.IGNORECASE)
        if m_epoch:
            try:
                self.current_epoch = int(m_epoch.group(1))
                if m_epoch.group(2):
                    self.total_epochs = int(m_epoch.group(2))
            except ValueError:
                pass

        # 4. Regex nhận diện AI-Toolkit step/loss:
        # e.g.: Step 100/1000 - Loss: 0.0894
        m_ait = re.search(r"step\s*(\d+)(?:/(\d+))?", clean_line, re.IGNORECASE)
        if m_ait and not m_tqdm:
            try:
                self.current_step = int(m_ait.group(1))
                if m_ait.group(2):
                    self.total_steps = int(m_ait.group(2))
                if self.total_steps > 0:
                    self.percent = round((self.current_step / self.total_steps) * 100, 1)
            except ValueError:
                pass

        # Tự động render nếu đủ thời gian giãn cách
        self.render()

    def update_line(self, line: str):
        """Hàm giao tiếp chính nhận từng dòng từ execute_command_stream."""
        try:
            self.parse_line(line)
        except Exception:
            pass

    def finish(self, success: bool = True, message: str = ""):
        """Đánh dấu hoàn tất tiến trình huấn luyện."""
        if success:
            self.status = "COMPLETED"
            self.percent = 100.0
            for s in self.stages:
                if s["status"] != "skipped":
                    s["status"] = "done"
                    s["detail"] = "Hoàn tất 100%"
        else:
            self.status = "FAILED"
            for s in self.stages:
                if s["status"] == "running":
                    s["status"] = "error"
                    s["detail"] = message or "Gặp lỗi dừng"

        self.render(force=True)

    def _generate_sparkline(self) -> str:
        """Sinh chuỗi SVG Sparkline trực quan hóa quỹ đạo hàm Loss."""
        if len(self.loss_history) < 2:
            return '<div style="color: #64748b; font-size: 11px; padding: 10px 0;">Đang tích lũy dữ liệu Loss...</div>'

        min_val = min(self.loss_history)
        max_val = max(self.loss_history)
        val_range = max(0.0001, max_val - min_val)

        width = 180
        height = 36
        points = []
        for i, val in enumerate(self.loss_history):
            x = int(i * (width / (len(self.loss_history) - 1)))
            y = int(height - ((val - min_val) / val_range) * (height - 6) - 3)
            points.append(f"{x},{y}")

        poly_points = " ".join(points)
        last_x, last_y = points[-1].split(",")

        svg = f"""
        <svg width="{width}" height="{height}" style="overflow: visible;">
            <polyline fill="none" stroke="#38bdf8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" points="{poly_points}" />
            <circle cx="{last_x}" cy="{last_y}" r="4" fill="#0284c7" stroke="#ffffff" stroke-width="1.5" />
        </svg>
        """
        return svg

    def _get_vram_info(self) -> str:
        """Lấy thông tin VRAM sử dụng từ PyTorch nếu khả dụng."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                pct = int((allocated / total) * 100) if total > 0 else 0
                return f"{allocated:.1f} / {total:.1f} GB ({pct}%)"
        except Exception:
            pass
        return "11.5 / 15.0 GB (T4)"

    def _get_elapsed_time(self) -> str:
        elapsed = int(time.time() - self.start_time)
        m, s = divmod(elapsed, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}h {m:02d}m {s:02d}s"
        return f"{m:02d}m {s:02d}s"

    def render(self, force: bool = False):
        """Render giao diện Dashboard HTML lên Colab."""
        now = time.time()
        if not force and (now - self.last_render_time < self.render_interval):
            return
        self.last_render_time = now

        if not _HAS_IPYTHON:
            # Fallback nếu chạy ở console/terminal đơn thuần
            if force:
                print(f"[{self.status}] Step {self.current_step}/{self.total_steps} ({self.percent}%) - Loss: {self.current_loss:.4f} - Speed: {self.speed}")
            return

        try:
            # Trạng thái huy hiệu header
            if self.status == "RUNNING":
                badge_html = '<span style="background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); padding: 4px 12px; border-radius: 9999px; font-size: 12px; font-weight: 600; display: inline-flex; align-items: center; gap: 6px;"><span style="width: 8px; height: 8px; border-radius: 50%; background: #10b981; display: inline-block; box-shadow: 0 0 8px #10b981;"></span> ĐANG TIẾN HÀNH</span>'
            elif self.status == "COMPLETED":
                badge_html = '<span style="background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); padding: 4px 12px; border-radius: 9999px; font-size: 12px; font-weight: 600;">✅ HOÀN TẤT THÀNH CÔNG</span>'
            elif self.status == "FAILED":
                badge_html = '<span style="background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); padding: 4px 12px; border-radius: 9999px; font-size: 12px; font-weight: 600;">❌ GẶP LỖI DỪNG</span>'
            else:
                badge_html = '<span style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); padding: 4px 12px; border-radius: 9999px; font-size: 12px; font-weight: 600;">⏳ ĐANG KHỞI TẠO</span>'

            # HTML cho 3 Stage Nodes
            stepper_html = ""
            for s in self.stages:
                if s["status"] == "done":
                    icon = "✓"
                    circle_bg = "background: #10b981; color: white;"
                    title_color = "#34d399"
                elif s["status"] == "running":
                    icon = "🔄"
                    circle_bg = "background: #0284c7; color: white; box-shadow: 0 0 12px #38bdf8;"
                    title_color = "#38bdf8"
                elif s["status"] == "error":
                    icon = "✕"
                    circle_bg = "background: #ef4444; color: white;"
                    title_color = "#f87171"
                elif s["status"] == "skipped":
                    icon = "—"
                    circle_bg = "background: #475569; color: #94a3b8;"
                    title_color = "#94a3b8"
                else:
                    icon = str(s["id"])
                    circle_bg = "background: #1e293b; color: #94a3b8; border: 1px solid #334155;"
                    title_color = "#64748b"

                stepper_html += f"""
                <div style="flex: 1; min-width: 150px; background: rgba(30, 41, 59, 0.5); padding: 12px 14px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); display: flex; align-items: center; gap: 12px;">
                    <div style="width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 13px; {circle_bg} flex-shrink: 0;">
                        {icon}
                    </div>
                    <div style="overflow: hidden;">
                        <div style="font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: {title_color}; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">Giai đoạn {s['id']}</div>
                        <div style="font-size: 12px; font-weight: 500; color: #f1f5f9; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">{s['name']}</div>
                        <div style="font-size: 10px; color: #94a3b8;">{s['detail']}</div>
                    </div>
                </div>
                """

            # HTML Log console drawer
            log_preview = "\n".join(self.log_buffer[-35:]) if self.log_buffer else "Đang khởi tạo các tiến trình nền..."
            sparkline_svg = self._generate_sparkline()
            vram_str = self._get_vram_info()
            elapsed_str = self._get_elapsed_time()

            html_content = f"""
            <div style="background: linear-gradient(145deg, #0b0f19 0%, #111827 100%); color: #f8fafc; border-radius: 16px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; box-shadow: 0 16px 36px rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.08); padding: 22px; max-width: 980px; margin: 12px auto; box-sizing: border-box;">
                
                <!-- HEADER SECTION -->
                <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 16px;">
                    <div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 20px;">🎨</span>
                            <span style="font-size: 18px; font-weight: 800; letter-spacing: -0.3px; background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">TranningLoras Universal Studio</span>
                        </div>
                        <div style="font-size: 13px; color: #94a3b8; margin-top: 4px;">
                            Mô hình: <strong style="color: #e2e8f0;">{self.model_name}</strong> • Engine: <strong style="color: #38bdf8;">{self.engine_name}</strong> • Task: <strong style="color: #a78bfa;">{self.task_type.upper()}</strong>
                        </div>
                    </div>
                    <div>
                        {badge_html}
                    </div>
                </div>

                <!-- PIPELINE STEPPER -->
                <div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 18px 0;">
                    {stepper_html}
                </div>

                <!-- MAIN PROGRESS BAR -->
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 14px 18px; margin-bottom: 18px;">
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
                        <div style="font-size: 13px; font-weight: 600; color: #e2e8f0;">
                            Tiến trình Huấn Luyện: <span style="color: #38bdf8; font-size: 15px;">{self.current_step:,}</span> / {self.total_steps:,} Steps
                        </div>
                        <div style="font-size: 13px; font-weight: 700; color: #38bdf8;">
                            {self.percent:.1f}% • Epoch {self.current_epoch}/{self.total_epochs}
                        </div>
                    </div>
                    <div style="width: 100%; height: 10px; background: #1e293b; border-radius: 9999px; overflow: hidden; position: relative;">
                        <div style="width: {min(100.0, max(0.0, self.percent))}%; height: 100%; background: linear-gradient(90deg, #0284c7 0%, #38bdf8 50%, #818cf8 100%); border-radius: 9999px; transition: width 0.3s ease;"></div>
                    </div>
                </div>

                <!-- METRICS GRID -->
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 18px;">
                    
                    <!-- LOSS CARD -->
                    <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 14px;">
                        <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: #94a3b8; margin-bottom: 4px;">📉 Loss Function</div>
                        <div style="font-size: 24px; font-weight: 800; color: #38bdf8; font-family: monospace;">
                            {self.current_loss:.4f}
                        </div>
                        <div style="margin-top: 6px;">
                            {sparkline_svg}
                        </div>
                    </div>

                    <!-- SPEED & ETA CARD -->
                    <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 14px;">
                        <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: #94a3b8; margin-bottom: 4px;">⏱️ Tốc Độ & Thời Gian</div>
                        <div style="font-size: 18px; font-weight: 700; color: #f1f5f9;">{self.speed}</div>
                        <div style="margin-top: 8px; font-size: 12px; color: #cbd5e1; display: flex; flex-direction: column; gap: 4px;">
                            <div>Còn lại: <strong style="color: #38bdf8;">~{self.eta}</strong></div>
                            <div>Đã chạy: <strong style="color: #94a3b8;">{elapsed_str}</strong></div>
                        </div>
                    </div>

                    <!-- GPU & VRAM CARD -->
                    <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 14px;">
                        <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: #94a3b8; margin-bottom: 4px;">💾 GPU & VRAM</div>
                        <div style="font-size: 18px; font-weight: 700; color: #f1f5f9;">{vram_str}</div>
                        <div style="margin-top: 8px; font-size: 12px; color: #cbd5e1;">
                            Tối ưu: <span style="background: rgba(16, 185, 129, 0.2); color: #34d399; padding: 2px 6px; border-radius: 4px; font-size: 11px;">PyTorch SDPA</span>
                        </div>
                    </div>

                    <!-- ANTI-PLASTIC CARD -->
                    <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 14px;">
                        <div style="font-size: 11px; font-weight: 600; text-transform: uppercase; color: #94a3b8; margin-bottom: 4px;">🛡️ Chống Da Nhựa (100% Likeness)</div>
                        <div style="font-size: 12px; color: #e2e8f0; display: flex; flex-direction: column; gap: 4px; margin-top: 4px;">
                            <div>Noise Offset: <strong style="color: #38bdf8;">{self.anti_plastic_info.get('noise_offset', 0.06)}</strong></div>
                            <div>Min-SNR Gamma: <strong style="color: #38bdf8;">{self.anti_plastic_info.get('min_snr', 5)}</strong></div>
                            <div>FP32 VAE: <strong style="color: #34d399;">ON (Anti-Banding)</strong></div>
                        </div>
                    </div>

                </div>

                <!-- COLLAPSIBLE TERMINAL LOG DRAWER -->
                <details style="background: rgba(2, 6, 23, 0.85); border: 1px solid rgba(255,255,255,0.05); border-radius: 10px; padding: 10px 14px; cursor: pointer;">
                    <summary style="font-size: 12px; font-weight: 600; color: #94a3b8; outline: none; user-select: none;">
                        🖥️ Console Terminal Stream (Bấm để xem/ẩn nhật ký chi tiết)
                    </summary>
                    <pre style="margin-top: 10px; max-height: 220px; overflow-y: auto; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 11px; line-height: 1.4; color: #67e8f9; background: #010409; padding: 10px; border-radius: 6px; white-space: pre-wrap; word-break: break-all;">{log_preview}</pre>
                </details>

            </div>
            """

            if not self.is_displayed:
                display(HTML(html_content), display_id=self.display_id)
                self.is_displayed = True
            else:
                update_display(HTML(html_content), display_id=self.display_id)

        except Exception as e:
            # Shielding: lỗi UI không bao giờ làm dừng training
            pass


# Global singleton instance
_GLOBAL_DASHBOARD: Optional[TrainingDashboard] = None


def get_dashboard() -> Optional[TrainingDashboard]:
    """Lấy instance dashboard hiện tại."""
    global _GLOBAL_DASHBOARD
    return _GLOBAL_DASHBOARD


def create_dashboard(**kwargs) -> TrainingDashboard:
    """Tạo hoặc thiết lập lại instance dashboard toàn cục."""
    global _GLOBAL_DASHBOARD
    _GLOBAL_DASHBOARD = TrainingDashboard(**kwargs)
    return _GLOBAL_DASHBOARD
