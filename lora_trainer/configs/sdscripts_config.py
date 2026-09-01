"""
Kohya sd-scripts Configuration Builder
Sinh cấu hình Dataset TOML và tham số dòng lệnh CLI cho các kiến trúc SDXL (Pony, Illustrious), SD 1.5 và SD 3.5.
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


def simple_sd_toml_dump(data: Dict[str, Any]) -> str:
    """Sinh chuỗi TOML cho sd-scripts."""
    lines = ["[general]"]
    if "general" in data:
        for k, v in data["general"].items():
            if isinstance(v, bool):
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
                if k == "subsets":
                    continue
                if isinstance(v, bool):
                    lines.append(f"{k} = {str(v).lower()}")
                elif isinstance(v, (int, float)):
                    lines.append(f"{k} = {v}")
                else:
                    lines.append(f'{k} = "{v}"')
            lines.append("")

            for sub in ds.get("subsets", []):
                lines.append("  [[datasets.subsets]]")
                for sk, sv in sub.items():
                    if isinstance(sv, bool):
                        lines.append(f"  {sk} = {str(sv).lower()}")
                    elif isinstance(sv, (int, float)):
                        lines.append(f"  {sk} = {sv}")
                    else:
                        lines.append(f'  {sk} = "{sv}"')
                lines.append("")

    return "\n".join(lines)


def safe_makedirs(path: str) -> None:
    """Tạo thư mục an toàn, bỏ qua lỗi nếu không có quyền hệ thống."""
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass


class SdScriptsConfigBuilder:
    """Xây dựng cấu hình TOML và CLI cho Kohya sd-scripts."""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        output_dir: str,
        output_name: str,
        arch: str = "sdxl",
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.output_dir = output_dir
        self.output_name = output_name
        self.arch = arch
        safe_makedirs(output_dir)

    def build_dataset_toml(
        self,
        dataset_path: str,
        image_folders: List[Dict[str, Any]],
        resolution: int = 1024,
        enable_bucket: bool = True,
        min_bucket_reso: int = 256,
        max_bucket_reso: int = 2048,
    ) -> str:
        """Sinh tệp dataset TOML chuẩn cho sd-scripts."""
        subsets = []
        for folder in image_folders:
            subsets.append({
                "image_dir": folder.get("path"),
                "num_repeats": folder.get("repeats", 1),
                "caption_extension": ".txt",
            })

        data = {
            "general": {
                "enable_bucket": enable_bucket,
                "caption_extension": ".txt",
                "shuffle_caption": True,
                "keep_tokens": 1,
                "bucket_reso_steps": 64,
                "bucket_no_upscale": False,
                "min_bucket_reso": min_bucket_reso,
                "max_bucket_reso": max_bucket_reso,
            },
            "datasets": [
                {
                    "resolution": resolution,
                    "subsets": subsets,
                }
            ],
        }

        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        if toml:
            with open(dataset_path, "w", encoding="utf-8") as f:
                toml.dump(data, f)
        else:
            with open(dataset_path, "w", encoding="utf-8") as f:
                f.write(simple_sd_toml_dump(data))

        return dataset_path

    def build_train_args(
        self,
        dataset_config_path: str,
        learning_rate: float = 1e-4,
        text_encoder_lr: Optional[float] = 5e-5,
        optimizer_type: str = "AdamW8bit",
        lr_scheduler: str = "cosine_with_restarts",
        network_dim: int = 32,
        network_alpha: int = 16,
        max_train_epochs: int = 10,
        save_every_n_epochs: int = 1,
        batch_size: int = 1,
        mixed_precision: str = "fp16",
        save_precision: str = "fp16",
        sample_prompt_file: Optional[str] = None,
        sample_every_n_steps: Optional[int] = 200,
        vae_path: Optional[str] = None,
        gradient_checkpointing: bool = True,
        cache_latents: bool = True,
        cache_latents_to_disk: bool = True,
        cache_text_encoder_outputs: bool = False,
        network_module: str = "networks.lora",
    ) -> str:
        """Sinh chuỗi tham số dòng lệnh CLI cho sd-scripts."""
        # Chọn script tương ứng
        if self.arch == "sdxl":
            script_name = "sdxl_train_network.py"
        elif self.arch == "sd35" or self.arch == "sd3":
            script_name = "sd3_train_network.py"
        else:
            script_name = "train_network.py"

        args = [
            "accelerate launch",
            f"--num_cpu_threads_per_process 4",
            script_name,
            f"--pretrained_model_name_or_path '{self.model_path}'",
            f"--dataset_config '{dataset_config_path}'",
            f"--output_dir '{self.output_dir}'",
            f"--output_name '{self.output_name}'",
            f"--save_model_as safetensors",
            f"--network_module {network_module}",
            f"--network_dim {network_dim}",
            f"--network_alpha {network_alpha}",
            f"--learning_rate {learning_rate}",
            f"--optimizer_type {optimizer_type}",
            f"--lr_scheduler {lr_scheduler}",
            f"--max_train_epochs {max_train_epochs}",
            f"--save_every_n_epochs {save_every_n_epochs}",
            f"--train_batch_size {batch_size}",
            f"--mixed_precision {mixed_precision}",
            f"--save_precision {save_precision}",
        ]

        if text_encoder_lr and not cache_text_encoder_outputs:
            args.append(f"--text_encoder_lr {text_encoder_lr}")

        if vae_path and os.path.exists(vae_path):
            args.append(f"--vae '{vae_path}'")

        if gradient_checkpointing:
            args.append("--gradient_checkpointing")

        if cache_latents:
            args.append("--cache_latents")
            if cache_latents_to_disk:
                args.append("--cache_latents_to_disk")

        if cache_text_encoder_outputs:
            args.append("--cache_text_encoder_outputs")
            args.append("--cache_text_encoder_outputs_to_disk")

        if sample_prompt_file and os.path.exists(sample_prompt_file) and sample_every_n_steps:
            args.append(f"--sample_prompts '{sample_prompt_file}'")
            args.append(f"--sample_every_n_steps {sample_every_n_steps}")
            args.append(f"--sample_sampler euler_a")

        return " ".join(args)
