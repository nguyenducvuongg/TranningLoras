"""
Ostris AI-Toolkit Configuration Builder
Sinh cấu hình YAML chuyên dụng cho Ostris AI-Toolkit (FLUX.1, Kontext, Inpainting, SDXL, DoRA/LoRA).
"""

import os
from typing import Dict, List, Any, Optional
import yaml
from ..core.model_registry import get_model_info


def safe_makedirs(path: str) -> None:
    """Tạo thư mục an toàn, bỏ qua lỗi nếu không có quyền hệ thống."""
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass


class ToolkitConfigBuilder:
    """Xây dựng cấu hình YAML hoàn chỉnh cho AI-Toolkit."""

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
        """Sinh file YAML cấu hình hoàn chỉnh cho AI-Toolkit."""
        datasets_cfg = []
        for ds in dataset_folders:
            item: Dict[str, Any] = {
                "folder_path": ds.get("path"),
                "caption_ext": ds.get("caption_ext", "txt"),
                "caption_dropout_rate": ds.get("caption_dropout_rate", 0.05),
                "cache_latents_to_disk": cache_latents_to_disk,
                "resolution": ds.get("resolution", self.model_info.get("default_resolution", [1024])),
            }
            if ds.get("control_path"):
                item["control_path"] = ds["control_path"]
            datasets_cfg.append(item)

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

        arch_name = self.model_info.get("toolkit_arch", self.model_info.get("arch", "flux"))

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
                "dtype": "bfloat16",
                "save_every": save_every,
                "max_step_saves_to_keep": 4,
                "push_to_hub": False,
            },
            "datasets": datasets_cfg,
            "train": {
                "batch_size": batch_size,
                "steps": steps,
                "gradient_accumulation_steps": 1,
                "train_unet": True,
                "train_text_encoder": False,
                "gradient_checkpointing": True,
                "noise_scheduler": "flowmatch",
                "optimizer": "adamw8bit",
                "lr": learning_rate,
                "lr_scheduler": lr_scheduler,
                "ema_config": {
                    "use_ema": True,
                    "ema_decay": 0.99,
                },
                "dtype": "bfloat16",
            },
            "model": {
                "name_or_path": self.model_info.get("name_or_path", self.model_name),
                "is_flux": "flux" in arch_name,
                "quantize": quantize,
                "low_vram": low_vram,
            },
            "sample": {
                "sampler": "flowmatch",
                "sample_every": sample_every,
                "width": width,
                "height": height,
                "prompts": samples_list,
                "neg": "",
                "seed": 42,
                "walk_seed": True,
                "guidance_scale": 3.5,
                "sample_steps": 20,
            },
        }

        if wandb_api_key:
            process_config["wandb"] = {
                "project": "TranningLoras",
                "entity": None,
                "api_key": wandb_api_key,
            }

        full_config = {
            "job": "extension",
            "config": {
                "name": self.output_name,
                "process": [process_config],
            },
        }

        os.makedirs(os.path.dirname(save_yaml_path), exist_ok=True)
        with open(save_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(full_config, f, sort_keys=False)

        return save_yaml_path
