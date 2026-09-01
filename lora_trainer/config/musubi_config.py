"""
Musubi-Tuner Configuration Builder
Sinh cấu hình dataset TOML, lệnh Pre-caching và tham số Accelerate CLI cho Kohya Musubi-Tuner.
"""

import os
from typing import Dict, List, Any, Optional, Union

try:
    import toml
except ImportError:
    try:
        import tomli_w as toml
    except ImportError:
        toml = None

from .model_registry import get_model_info, VAE_REGISTRY, TEXT_ENCODER_REGISTRY


def safe_makedirs(path: str) -> None:
    """Tạo thư mục an toàn, bỏ qua lỗi nếu không có quyền hệ thống."""
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass


def simple_toml_dump(data: Dict[str, Any]) -> str:
    """Tự sinh chuỗi TOML cơ bản khi chưa cài đặt thư viện toml."""
    lines = []
    
    # Sections
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
    """
    Chuyển đổi một dictionary thành chuỗi tham số dòng lệnh (CLI arguments).
    Nếu value là True, chỉ render cờ flag.
    Nếu value là False hoặc None, bỏ qua.
    Nếu value là chuỗi/số, render `--key value`.
    """
    args = []
    for key, value in config_dict.items():
        if value is False or value is None or value == "":
            continue
        
        # Nếu là câu lệnh/script đầu tiên không có tiền tố '--'
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
        safe_makedirs(weights_dir)

    def get_local_path(self, resource_key: str) -> str:
        """Trả về đường dẫn file đã tải về local của VAE/TextEncoder/Model."""
        filename = f"{resource_key}.safetensors"
        if "pth" in resource_key or "wan" in resource_key:
            filename = f"{resource_key}.pth"
        return os.path.join(self.weights_dir, filename)

    def build_dataset_toml(
        self,
        dataset_path: str,
        resolution: List[int],
        image_folders: Optional[List[Dict[str, Any]]] = None,
        video_folders: Optional[List[Dict[str, Any]]] = None,
        enable_bucket: bool = True,
        bucket_no_upscale: bool = True,
        resize_control: bool = True,
    ) -> str:
        """
        Sinh file dataset.toml cho Musubi-tuner hỗ trợ đa thư mục ảnh và video.
        """
        data_config: Dict[str, Any] = {
            "general": {
                "resolution": resolution,
                "enable_bucket": enable_bucket,
                "bucket_no_upscale": bucket_no_upscale,
            },
            "datasets": [],
        }

        # Thêm các tập dữ liệu hình ảnh
        if image_folders:
            for folder in image_folders:
                img_dir = folder.get("path")
                ctrl_dir = folder.get("control_path", None)
                repeats = folder.get("repeats", 1)
                
                cache_dir = os.path.join(
                    self.cache_base_dir, f"latents_{self.model_info['arch']}_{os.path.basename(img_dir)}"
                )
                safe_makedirs(cache_dir)

                ds_item: Dict[str, Any] = {
                    "image_directory": img_dir,
                    "cache_directory": cache_dir,
                    "num_repeats": repeats,
                }
                if ctrl_dir and os.path.exists(ctrl_dir):
                    ds_item["control_directory"] = ctrl_dir
                    if not resize_control and "kontext" in self.model_info["arch"]:
                        ds_item["flux_kontext_no_resize_control"] = True
                    if not resize_control and "qwen_image_edit" in self.model_info["arch"]:
                        ds_item["qwen_image_edit_no_resize_control"] = True

                data_config["datasets"].append(ds_item)

        # Thêm các tập dữ liệu video
        if video_folders:
            for folder in video_folders:
                vid_dir = folder.get("path")
                repeats = folder.get("repeats", 1)
                frame_extraction = folder.get("frame_extraction", "chunk")
                target_frames = folder.get("target_frames", [25])
                frame_stride = folder.get("frame_stride", 1)
                frame_sample = folder.get("frame_sample", 1)
                max_frames = folder.get("max_frames", 33)

                cache_dir = os.path.join(
                    self.cache_base_dir, f"cachevideo_{self.model_info['arch']}_{os.path.basename(vid_dir)}"
                )
                safe_makedirs(cache_dir)

                ds_item = {
                    "video_directory": vid_dir,
                    "cache_directory": cache_dir,
                    "num_repeats": repeats,
                    "frame_extraction": frame_extraction,
                    "target_frames": target_frames,
                    "frame_stride": frame_stride,
                    "frame_sample": frame_sample,
                    "max_frames": max_frames,
                }
                data_config["datasets"].append(ds_item)

        safe_makedirs(os.path.dirname(dataset_path))
        if toml is not None and hasattr(toml, "dump"):
            with open(dataset_path, "w", encoding="utf-8") as f:
                toml.dump(data_config, f)
        else:
            with open(dataset_path, "w", encoding="utf-8") as f:
                f.write(simple_toml_dump(data_config))

        return dataset_path

    def build_cache_latents_args(
        self,
        dataset_config_path: str,
        vae_path: str,
        clip_vision_path: Optional[str] = None,
    ) -> str:
        """Sinh CLI arguments cho bước cache latents."""
        arch = self.model_info["arch"]
        
        # Chọn script cache tương ứng với kiến trúc
        if "wan" in arch:
            script = "wan_cache_latents.py"
        elif "kontext" in arch:
            script = "src/musubi_tuner/flux_kontext_cache_latents.py"
        elif "flux2" in arch:
            script = "src/musubi_tuner/flux_2_cache_latents.py"
        elif "qwen" in arch:
            script = "src/musubi_tuner/qwen_image_cache_latents.py"
        elif "z_image" in arch:
            script = "src/musubi_tuner/zimage_cache_latents.py"
        elif "krea" in arch:
            script = "src/musubi_tuner/krea2_cache_latents.py"
        else:
            script = "src/musubi_tuner/cache_latents.py"

        cfg: Dict[str, Any] = {
            "python": True,
            script: True,
            "--dataset_config": f'"{dataset_config_path}"',
            "--vae": f'"{vae_path}"',
        }

        if "wan" in arch and self.model_info.get("supports_i2v", False):
            cfg["--i2v"] = True
            if clip_vision_path:
                cfg["--clip"] = f'"{clip_vision_path}"'

        if "model_version" in self.model_info:
            cfg["--model_version"] = self.model_info["model_version"]

        return dict_to_cli_args(cfg)

    def build_cache_text_encoder_args(
        self,
        dataset_config_path: str,
        text_encoder1_path: str,
        text_encoder2_path: Optional[str] = None,
        batch_size: int = 16,
    ) -> str:
        """Sinh CLI arguments cho bước cache text encoder outputs."""
        arch = self.model_info["arch"]

        if "wan" in arch:
            script = "wan_cache_text_encoder_outputs.py"
            cfg = {
                "python": True,
                script: True,
                "--dataset_config": f'"{dataset_config_path}"',
                "--t5": f'"{text_encoder1_path}"',
                "--batch_size": batch_size,
            }
            return dict_to_cli_args(cfg)

        if "kontext" in arch:
            script = "src/musubi_tuner/flux_kontext_cache_text_encoder_outputs.py"
        elif "flux2" in arch:
            script = "src/musubi_tuner/flux_2_cache_text_encoder_outputs.py"
        elif "qwen" in arch:
            script = "src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py"
        elif "z_image" in arch:
            script = "src/musubi_tuner/zimage_cache_text_encoder_outputs.py"
        elif "krea" in arch:
            script = "src/musubi_tuner/krea2_cache_text_encoder_outputs.py"
        else:
            script = "src/musubi_tuner/cache_text_encoder_outputs.py"

        cfg = {
            "python": True,
            script: True,
            "--dataset_config": f'"{dataset_config_path}"',
            "--batch_size": batch_size,
        }

        if text_encoder2_path:
            cfg["--text_encoder1"] = f'"{text_encoder1_path}"'
            cfg["--text_encoder2"] = f'"{text_encoder2_path}"'
        else:
            cfg["--text_encoder"] = f'"{text_encoder1_path}"'

        if "model_version" in self.model_info:
            cfg["--model_version"] = self.model_info["model_version"]

        return dict_to_cli_args(cfg)

    def build_train_args(
        self,
        dataset_config_path: str,
        dit_model_path: str,
        learning_rate: float = 1e-4,
        optimizer_type: str = "adamw8bit",
        lr_scheduler: str = "constant",
        network_dim: int = 32,
        network_alpha: int = 16,
        max_train_epochs: int = 5,
        max_train_steps: int = 0,
        save_every_n_epochs: int = 1,
        save_every_n_steps: int = 0,
        timestep_sampling: Optional[str] = None,
        min_timestep: int = 0,
        max_timestep: int = 1000,
        timestep_boundary: Optional[int] = None,
        fp8_base: bool = True,
        gradient_checkpointing: bool = True,
        sample_prompt_file: Optional[str] = None,
        sample_every_n_steps: int = 0,
        wandb_api_key: Optional[str] = None,
        resume_path: Optional[str] = None,
        seed: int = 42,
    ) -> str:
        """Sinh CLI arguments đầy đủ cho lệnh Accelerate launch."""
        train_script = self.model_info.get("musubi_train_script", "train_network.py")
        arch = self.model_info["arch"]

        # Chọn network module chuẩn xác theo từng kiến trúc
        if "wan" in arch:
            network_module = "networks.lora_wan"
            default_ts = "shift"
        elif "krea" in arch:
            network_module = "networks.lora_krea2"
            default_ts = "krea2_shift"
        elif "flux2" in arch:
            network_module = "networks.lora_flux_2"
            default_ts = "flux2_shift"
        elif "kontext" in arch:
            network_module = "networks.lora_flux"
            default_ts = "flux_shift"
        elif "qwen" in arch:
            network_module = "networks.lora_qwen_image"
            default_ts = "shift"
        elif "z_image" in arch:
            network_module = "networks.lora_zimage"
            default_ts = "shift"
        else:
            network_module = "networks.lora_flux"
            default_ts = "shift"

        effective_timestep_sampling = timestep_sampling or default_ts

        # Cấu hình Accelerate launch
        accel_cfg: Dict[str, Any] = {
            "accelerate launch": True,
            "--num_cpu_threads_per_process": 1,
            "--mixed_precision": "bf16",
            train_script: True,
            "--dit": f'"{dit_model_path}"',
            "--dataset_config": f'"{dataset_config_path}"',
            "--output_dir": f'"{self.output_dir}"',
            "--output_name": f'"{self.output_name}"',
            "--learning_rate": learning_rate,
            "--optimizer_type": optimizer_type,
            "--lr_scheduler": lr_scheduler,
            "--network_module": network_module,
            "--network_dim": network_dim,
            "--network_alpha": network_alpha,
            "--timestep_sampling": effective_timestep_sampling,
            "--min_timestep": min_timestep,
            "--max_timestep": max_timestep,
            "--discrete_flow_shift": self.model_info.get("discrete_flow_shift", 3.0),
            "--mixed_precision": "bf16",
            "--save_precision": "bf16",
            "--sdpa": True,
            "--split_attn": True,
            "--max_data_loader_n_workers": 2,
            "--persistent_data_loader_workers": True,
            "--weighting_scheme": "none",
            "--seed": seed,
        }

        if fp8_base and "flux2" not in arch and "krea" not in arch:
            accel_cfg["--fp8_base"] = True
        if "qwen" in arch:
            accel_cfg["--fp8_vl"] = True
        if gradient_checkpointing:
            accel_cfg["--gradient_checkpointing"] = True

        if max_train_steps > 0:
            accel_cfg["--max_train_steps"] = max_train_steps
        elif max_train_epochs > 0:
            accel_cfg["--max_train_epochs"] = max_train_epochs

        if save_every_n_steps > 0:
            accel_cfg["--save_every_n_steps"] = save_every_n_steps
        elif save_every_n_epochs > 0:
            accel_cfg["--save_every_n_epochs"] = save_every_n_epochs

        # Wan 2.2 Boundary parameter
        if "wan22" in arch:
            boundary = timestep_boundary if timestep_boundary is not None else self.model_info.get("default_boundary", 875)
            accel_cfg["--timestep_boundary"] = boundary

        # Validation Preview / Sampling
        if sample_prompt_file and sample_every_n_steps > 0:
            accel_cfg["--sample_prompts"] = f'"{sample_prompt_file}"'
            accel_cfg["--sample_every_n_steps"] = sample_every_n_steps
            accel_cfg["--sample_at_first"] = True

        # Weights & Biases Logging
        if wandb_api_key:
            accel_cfg["--log_with"] = "wandb"
            accel_cfg["--wandb_api_key"] = wandb_api_key
            accel_cfg["--wandb_run_name"] = f'"{self.output_name}_{self.model_name}"'

        # Resume state
        if resume_path:
            accel_cfg["--resume"] = f'"{resume_path}"'

        return dict_to_cli_args(accel_cfg)
