"""
Kohya Musubi-Tuner Configuration Builder
Sinh cấu hình Dataset TOML, lệnh Pre-caching (Latents & Text Encoders) và tham số Accelerate CLI
cho các dòng mô hình MMDiT & Video (Wan 2.1/2.2, FLUX.2 Klein, Qwen-Image, Z-Image, Krea2).
"""

import os
from typing import Dict, List, Any, Optional

try:
    import toml
except ImportError:
    try:
        import tomli_w as toml
    except ImportError:
        toml = None

from ..core.model_registry import get_model_info, VAE_REGISTRY, TEXT_ENCODER_REGISTRY


def simple_musubi_toml_dump(data: Dict[str, Any]) -> str:
    """Tự sinh chuỗi TOML cơ bản khi chưa cài đặt thư viện toml."""
    lines = []
    if "general" in data:
        lines.append("[general]")
        for k, v in data["general"].items():
            if isinstance(v, list):
                lines.append(f"{k} = {v}")
            elif isinstance(v, bool):
                lines.append(f"{k} = {str(v).lower()}")
            elif isinstance(v, (int, float)):
                lines.append(f"{k} = {v}")
            else:
                lines.append(f'{k} = "{v}"')
        lines.append("")

    if "datasets" in data:
        for ds in data["datasets"]:
            lines.append("[[datasets]]")
            for k, v in ds.items():
                if v is None:
                    continue
                if isinstance(v, list):
                    lines.append(f"{k} = {v}")
                elif isinstance(v, bool):
                    lines.append(f"{k} = {str(v).lower()}")
                elif isinstance(v, (int, float)):
                    lines.append(f"{k} = {v}")
                else:
                    lines.append(f'{k} = "{v}"')
            lines.append("")

    return "\n".join(lines)


def dict_to_cli_args(config_dict: Dict[str, Any]) -> str:
    """Chuyển đổi một dictionary thành chuỗi tham số dòng lệnh (CLI arguments)."""
    args = []
    for key, value in config_dict.items():
        if value is False or value is None or value == "":
            continue

        if not key.startswith("--") and not key.startswith("-"):
            if value is True:
                args.append(key)
            else:
                args.append(f"{key} {value}")
            continue

        if value is True:
            args.append(key)
        else:
            args.append(f"{key} {value}")

    return " ".join(args)


def ensure_dataset_captions(image_folders: List[Dict[str, Any]], default_caption: str = "") -> None:
    """Đảm bảo mọi ảnh trong tập huấn luyện đều có tệp .txt tương ứng."""
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"}
    for folder in image_folders:
        dir_path = folder.get("path")
        if not dir_path or not os.path.exists(dir_path):
            continue
        try:
            for fname in os.listdir(dir_path):
                fpath = os.path.join(dir_path, fname)
                if not os.path.isfile(fpath):
                    continue
                ext = os.path.splitext(fname)[1]
                if ext in valid_exts:
                    txt_path = os.path.splitext(fpath)[0] + ".txt"
                    if not os.path.exists(txt_path):
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(default_caption)
        except Exception:
            pass


def safe_makedirs(path: str) -> None:
    """Tạo thư mục an toàn, bỏ qua lỗi nếu không có quyền hệ thống."""
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass


class MusubiConfigBuilder:
    """Xây dựng cấu hình hoàn chỉnh cho Kohya Musubi-Tuner."""

    def __init__(
        self,
        model_name: str,
        output_dir: str,
        output_name: str,
        cache_base_dir: str = "/content/cache",
        weights_dir: str = "/content/models",
    ):
        self.model_name = model_name
        self.model_info = get_model_info(model_name)
        self.output_dir = output_dir
        self.output_name = output_name
        self.cache_base_dir = cache_base_dir
        self.weights_dir = weights_dir
        safe_makedirs(output_dir)
        safe_makedirs(cache_base_dir)

    def build_dataset_toml(
        self,
        dataset_path: str,
        resolution: Optional[List[int]] = None,
        image_folders: Optional[List[Dict[str, Any]]] = None,
        caption_extension: str = ".txt",
        batch_size: int = 1,
    ) -> str:
        """Sinh tệp dataset TOML cho Image DiT models trong Musubi-Tuner."""
        res = resolution if resolution else self.model_info.get("default_resolution", [1024, 1024])
        if len(res) == 1:
            res = [res[0], res[0]]

        general_cfg = {
            "resolution": res,
            "caption_extension": caption_extension,
            "batch_size": batch_size,
            "enable_bucket": True,
            "bucket_no_upscale": False,
        }

        datasets_list = []

        if image_folders:
            ensure_dataset_captions(image_folders)
            for f in image_folders:
                item: Dict[str, Any] = {
                    "image_dir": f.get("path"),
                    "num_repeats": f.get("repeats", 1),
                }
                if f.get("control_path"):
                    item["control_image_dir"] = f["control_path"]
                if f.get("resolution"):
                    item["resolution"] = f["resolution"]
                datasets_list.append(item)

        data = {
            "general": general_cfg,
            "datasets": datasets_list,
        }

        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        if toml:
            with open(dataset_path, "w", encoding="utf-8") as f:
                toml.dump(data, f)
        else:
            with open(dataset_path, "w", encoding="utf-8") as f:
                f.write(simple_musubi_toml_dump(data))

        return dataset_path

    def build_cache_latents_args(
        self,
        dataset_config_path: str,
        vae_path: str,
        clip_vision_path: Optional[str] = None,
        batch_size: int = 1,
    ) -> str:
        """Sinh câu lệnh Pre-cache VAE Latents cho Musubi."""
        arch = self.model_info.get("arch", "flux_kontext")
        if arch in ["flux_kontext", "flux"]:
            script = "src/musubi_tuner/flux_kontext_cache_latents.py" if arch == "flux_kontext" else "src/musubi_tuner/flux_cache_latents.py"
            return f"python {script} --dataset_config '{dataset_config_path}' --vae '{vae_path}' --batch_size {batch_size}"
        elif arch == "flux2":
            return f"python src/musubi_tuner/flux_2_cache_latents.py --dataset_config '{dataset_config_path}' --vae '{vae_path}' --batch_size {batch_size}"
        elif "qwen" in arch:
            return f"python src/musubi_tuner/qwen_image_cache_latents.py --dataset_config '{dataset_config_path}' --vae '{vae_path}' --batch_size {batch_size}"
        elif arch == "z_image":
            return f"python src/musubi_tuner/zimage_cache_latents.py --dataset_config '{dataset_config_path}' --vae '{vae_path}' --batch_size {batch_size}"
        elif arch == "krea2":
            return f"python src/musubi_tuner/krea2_cache_latents.py --dataset_config '{dataset_config_path}' --vae '{vae_path}' --batch_size {batch_size}"
        else:
            return f"python src/musubi_tuner/cache_latents.py --dataset_config '{dataset_config_path}' --vae '{vae_path}' --batch_size {batch_size}"

    def build_cache_text_encoder_outputs_args(
        self,
        dataset_config_path: str,
        text_encoder1_path: str,
        text_encoder2_path: Optional[str] = None,
        batch_size: int = 1,
    ) -> str:
        """Sinh câu lệnh Pre-cache Text Encoders cho Musubi."""
        arch = self.model_info.get("arch", "flux_kontext")
        if arch in ["flux_kontext", "flux"]:
            script = "src/musubi_tuner/flux_kontext_cache_text_encoder_outputs.py" if arch == "flux_kontext" else "src/musubi_tuner/flux_cache_text_encoder_outputs.py"
            cmd = f"python {script} --dataset_config '{dataset_config_path}' --clip_l '{text_encoder1_path}'"
            if text_encoder2_path:
                cmd += f" --t5xxl '{text_encoder2_path}'"
            return cmd
        elif arch == "flux2":
            return f"python src/musubi_tuner/flux_2_cache_text_encoder_outputs.py --dataset_config '{dataset_config_path}' --qwen_3 '{text_encoder1_path}'"
        elif "qwen" in arch:
            return f"python src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py --dataset_config '{dataset_config_path}' --text_encoder '{text_encoder1_path}'"
        elif arch == "z_image":
            return f"python src/musubi_tuner/zimage_cache_text_encoder_outputs.py --dataset_config '{dataset_config_path}' --text_encoder '{text_encoder1_path}'"
        elif arch == "krea2":
            return f"python src/musubi_tuner/krea2_cache_text_encoder_outputs.py --dataset_config '{dataset_config_path}' --text_encoder '{text_encoder1_path}'"
        else:
            return f"python src/musubi_tuner/cache_text_encoder_outputs.py --dataset_config '{dataset_config_path}' --text_encoder '{text_encoder1_path}'"

    # Alias để tương thích
    build_cache_text_encoder_args = build_cache_text_encoder_outputs_args

    def build_train_args(
        self,
        dataset_config_path: str,
        dit_model_path: str,
        learning_rate: float = 1e-4,
        optimizer_type: str = "adamw8bit",
        lr_scheduler: str = "constant",
        network_dim: int = 32,
        network_alpha: int = 16,
        max_train_epochs: int = 10,
        save_every_n_epochs: int = 1,
        mixed_precision: str = "bf16",
        sample_prompt_file: Optional[str] = None,
        sample_every_n_steps: Optional[int] = 200,
        timestep_boundary: Optional[int] = None,
        discrete_flow_shift: Optional[float] = None,
        gradient_checkpointing: bool = True,
        fp8_base: bool = True,
    ) -> str:
        """Sinh chuỗi tham số CLI kích hoạt huấn luyện Musubi qua Accelerate."""
        arch = self.model_info.get("arch", "flux_kontext")
        script = self.model_info.get("musubi_train_script", "src/musubi_tuner/flux_kontext_train_network.py")

        args = [
            "accelerate launch",
            f"--num_cpu_threads_per_process 4",
            script,
            f"--dataset_config '{dataset_config_path}'",
            f"--dit '{dit_model_path}'",
            f"--output_dir '{self.output_dir}'",
            f"--output_name '{self.output_name}'",
            f"--learning_rate {learning_rate}",
            f"--optimizer_type {optimizer_type}",
            f"--lr_scheduler {lr_scheduler}",
            f"--network_dim {network_dim}",
            f"--network_alpha {network_alpha}",
            f"--max_train_epochs {max_train_epochs}",
            f"--save_every_n_epochs {save_every_n_epochs}",
            f"--mixed_precision {mixed_precision}",
            f"--save_precision {mixed_precision}",
            "--sdpa",
        ]

        if arch in ["flux", "flux_kontext", "flux2"]:
            args.append("--network_module networks.lora_flux")
        elif "qwen" in arch:
            args.append("--network_module networks.lora_qwen")
        elif arch == "z_image":
            args.append("--network_module networks.lora_zimage")
        elif arch == "krea2":
            args.append("--network_module networks.lora_krea")
        else:
            args.append("--network_module networks.lora")

        if gradient_checkpointing:
            args.append("--gradient_checkpointing")

        if fp8_base:
            args.append("--fp8_base")

        flow_shift = discrete_flow_shift or self.model_info.get("discrete_flow_shift")
        if flow_shift is not None:
            args.append(f"--discrete_flow_shift {flow_shift}")

        boundary = timestep_boundary or self.model_info.get("default_boundary")
        if boundary is not None:
            args.append(f"--timestep_boundary {boundary}")

        if sample_prompt_file and os.path.exists(sample_prompt_file) and sample_every_n_steps:
            args.append(f"--sample_prompts '{sample_prompt_file}'")
            args.append(f"--sample_every_n_steps {sample_every_n_steps}")

        return " ".join(args)
