"""
AI-Toolkit Configuration Builder
Sinh cấu hình YAML chuyên dụng cho Ostris AI-Toolkit (FLUX.1, Kontext, Inpainting, Multi-Control).
"""

import os
from typing import Dict, List, Any, Optional
import yaml
from .model_registry import get_model_info


def safe_makedirs(path: str) -> None:
    """Tạo thư mục an toàn, bỏ qua lỗi nếu không có quyền hệ thống."""
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass


class ToolkitConfigBuilder:
    """Xây dựng cấu hình YAML cho AI-Toolkit."""

    def __init__(
        self,
        model_name: str,
        output_dir: str,
        output_name: str,
    ):
        self.model_name = model_name
        self.model_info = get_model_info(model_name)
        self.output_dir = output_dir
        self.output_name = output_name
        safe_makedirs(output_dir)

    def build_yaml_config(
        self,
        save_yaml_path: str,
        dataset_folders: List[Dict[str, Any]],
        steps: int = 1000,
        save_every: int = 250,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        lr_scheduler: str = "constant",
        linear_dim: int = 16,
        linear_alpha: int = 16,
        quantize: bool = True,
        low_vram: bool = False,
        cache_latents_to_disk: bool = True,
        sample_prompts: Optional[List[str]] = None,
        sample_every: int = 250,
        sample_resolution: Optional[List[int]] = None,
        wandb_api_key: Optional[str] = None,
        trigger_word: Optional[str] = None,
    ) -> str:
        """
        Sinh file YAML cấu hình hoàn chỉnh cho AI-Toolkit.
        """
        # Xây dựng danh sách datasets
        datasets_cfg = []
        for ds in dataset_folders:
            item: Dict[str, Any] = {
                "folder_path": ds.get("path"),
                "caption_ext": ds.get("caption_ext", "txt"),
                "caption_dropout_rate": ds.get("caption_dropout_rate", 0.05),
                "cache_latents_to_disk": cache_latents_to_disk,
                "resolution": ds.get("resolution", self.model_info.get("default_resolution", [1024])),
            }

            # Hỗ trợ multi-control
            if ds.get("control_path"):
                item["control_path"] = ds["control_path"]
            if ds.get("control_path_1"):
                item["control_path_1"] = ds["control_path_1"]
            if ds.get("control_path_2"):
                item["control_path_2"] = ds["control_path_2"]
            if ds.get("control_path_3"):
                item["control_path_3"] = ds["control_path_3"]

            datasets_cfg.append(item)

        # Cấu hình Sample validation
        if sample_resolution is None:
            sample_resolution = [1024, 1024]
        width = sample_resolution[0]
        height = sample_resolution[1] if len(sample_resolution) > 1 else sample_resolution[0]

        samples_list = []
        if sample_prompts:
            for p in sample_prompts:
                samples_list.append({"prompt": p})
        else:
            samples_list.append({"prompt": f"photo of {trigger_word if trigger_word else 'subject'}, high quality, 8k"})

        process_config: Dict[str, Any] = {
            "type": "diffusion_trainer",
            "training_folder": self.output_dir,
            "device": "cuda",
            "trigger_word": trigger_word,
            "performance_log_every": 10,
            "network": {
                "type": "lora",
                "linear": linear_dim,
                "linear_alpha": linear_alpha,
            },
            "save": {
                "dtype": "float16",
                "save_every": save_every,
                "max_step_saves_to_keep": 50,
                "save_format": "diffusers",
                "push_to_hub": False,
            },
            "datasets": datasets_cfg,
            "logging": {
                "use_wandb": True if wandb_api_key else False,
                "project_name": "colab_lora_trainer",
                "run_name": f"{self.output_name}_{self.model_info['arch']}",
                "log_every": 10,
            },
            "train": {
                "batch_size": batch_size,
                "steps": steps,
                "gradient_accumulation": 1,
                "train_unet": True,
                "train_text_encoder": False,
                "gradient_checkpointing": True,
                "noise_scheduler": "flowmatch",
                "optimizer": "adamw8bit",
                "timestep_type": "shift",
                "lr": learning_rate,
                "lr_scheduler": lr_scheduler,
                "dtype": "bf16",
                "disable_sampling": False if sample_every > 0 else True,
            },
            "model": {
                "name_or_path": self.model_info.get("name_or_path", self.model_name),
                "arch": self.model_info.get("arch", "flux"),
                "low_vram": low_vram,
                "quantize": quantize,
                "qtype": "qfloat8" if quantize else None,
                "quantize_te": quantize,
                "qtype_te": "qfloat8" if quantize else None,
            },
            "sample": {
                "sampler": "flowmatch",
                "sample_every": sample_every,
                "width": width,
                "height": height,
                "samples": samples_list,
                "seed": 42,
                "walk_seed": True,
                "guidance_scale": 3.5,
                "sample_steps": 20,
            },
        }

        full_config = {
            "job": "extension",
            "config": {
                "name": self.output_name,
                "process": [process_config],
            },
            "meta": {
                "name": self.output_name,
                "version": "1.0",
            },
        }

        safe_makedirs(os.path.dirname(save_yaml_path))
        with open(save_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)

        return save_yaml_path
