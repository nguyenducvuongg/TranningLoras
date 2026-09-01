"""
Universal Unified LoRA Trainer Facade
Lớp điều phối trung tâm tự động phân giải mô hình, tải trọng số, sinh cấu hình và kích hoạt Engine tối ưu nhất
(Kohya sd-scripts, Kohya Musubi-Tuner, hoặc Ostris AI-Toolkit).
"""

import os
from typing import Dict, Any, List, Optional, Union
from ..core.model_registry import get_model_info, get_preferred_engine
from ..core.hardware import detect_hardware_environment, setup_cuda_environment
from ..storage.model_fetcher import download_model_suite
from ..dataset.builder import build_dataset_list
from ..configs.sdscripts_config import SdScriptsConfigBuilder
from ..configs.musubi_config import MusubiConfigBuilder
from ..configs.toolkit_config import ToolkitConfigBuilder
from .sdscripts_engine import run_sdscripts_pipeline
from .musubi_engine import run_musubi_pipeline
from .toolkit_engine import run_toolkit_pipeline
from ..utils.prompt_sampler import get_random_sample_prompt
from ..utils.lora_converter import auto_convert_checkpoints


def run_unified_training(
    model_name: str,
    train_folders: str,
    output_dir: str = "/content/drive/MyDrive/TranningLorasData/outputs",
    output_name: str = "my_lora",
    control_folder: Optional[str] = None,
    resolution: Union[str, List[int]] = "1024,1024",
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    optimizer: str = "adamw8bit",
    lr_scheduler: str = "constant",
    network_dim: int = 32,
    network_alpha: int = 16,
    max_train_epochs: int = 10,
    save_every_n_epochs: int = 1,
    sample_every_n_steps: int = 200,
    sample_prompt: str = "",
    hf_token: Optional[str] = None,
    civitai_key: Optional[str] = None,
    wandb_key: Optional[str] = None,
    custom_model_url: Optional[str] = None,
    engine_override: Optional[str] = None,
    base_drive_dir: str = "/content/drive/MyDrive/TranningLorasData",
    weights_dir: str = "/content/models",
    cache_dir: str = "/content/cache",
    convert_to_comfy: bool = True,
) -> bool:
    """
    Hàm thực thi huấn luyện Universal:
    Nhận mọi thông số và tự động thực hiện từ A-Z một cách tối ưu nhất.
    """
    setup_cuda_environment()
    hw_info = detect_hardware_environment()

    # 1. Xử lý độ phân giải
    if isinstance(resolution, str):
        res_list = [int(x.strip()) for x in resolution.split(",") if x.strip()]
    else:
        res_list = list(resolution)
    if len(res_list) == 1:
        res_list = [res_list[0], res_list[0]]

    # 2. Xây dựng Dataset List
    datasets = build_dataset_list(train_folders, control_folder, resolution=res_list)
    if not datasets:
        raise ValueError(f"Không tìm thấy thư mục dữ liệu hợp lệ: {train_folders}")

    # 3. Phân giải Engine tối ưu
    model_info = get_model_info(model_name, custom_download_url=custom_model_url)
    engine_type = engine_override if engine_override else get_preferred_engine(model_name)
    arch = model_info.get("arch", "sdxl")

    print("\n" + "=" * 60)
    print(f"🎯 UNIVERSAL LORA TRAINER")
    print(f"📌 Mô hình: {model_name} (Kiến trúc: {arch.upper()})")
    print(f"⚡ Engine huấn luyện: {engine_type.upper()}")
    print(f"🚀 GPU: {hw_info['gpu_name']} | VRAM: {hw_info['vram_gb']} GB")
    print("=" * 60 + "\n")

    # 4. Tải trước các trọng số cần thiết
    weights = download_model_suite(
        model_name=model_name,
        weights_dir=weights_dir,
        hf_token=hf_token,
        civitai_key=civitai_key,
        base_drive_dir=base_drive_dir,
        custom_url=custom_model_url,
    )

    # 5. Xử lý Sample Prompt
    sample_txt_path = "/content/sample_prompt.txt"
    final_sample_prompt = sample_prompt.strip()
    if not final_sample_prompt:
        p, _, _ = get_random_sample_prompt(datasets[0]["path"], datasets[0].get("control_path"))
        final_sample_prompt = p
    with open(sample_txt_path, "w", encoding="utf-8") as f:
        f.write(f"{final_sample_prompt} --w {res_list[0]} --h {res_list[1]}\n")

    success = False

    # 6. Điều hướng Engine thực thi
    if engine_type == "sdscripts":
        builder = SdScriptsConfigBuilder(
            model_name=model_name,
            model_path=weights.get("dit", ""),
            output_dir=output_dir,
            output_name=output_name,
            arch=arch,
        )
        dataset_toml = "/content/sdscripts_dataset.toml"
        builder.build_dataset_toml(
            dataset_path=dataset_toml,
            image_folders=datasets,
            resolution=res_list[0],
        )
        train_cmd = builder.build_train_args(
            dataset_config_path=dataset_toml,
            learning_rate=learning_rate,
            optimizer_type=optimizer if optimizer != "adamw8bit" else "AdamW8bit",
            lr_scheduler=lr_scheduler,
            network_dim=network_dim,
            network_alpha=network_alpha,
            max_train_epochs=max_train_epochs,
            save_every_n_epochs=save_every_n_epochs,
            batch_size=batch_size,
            mixed_precision=hw_info["recommended_precision"],
            sample_prompt_file=sample_txt_path,
            sample_every_n_steps=sample_every_n_steps,
            vae_path=weights.get("vae"),
        )
        success = run_sdscripts_pipeline(train_cmd)

    elif engine_type == "musubi":
        builder = MusubiConfigBuilder(
            model_name=model_name,
            output_dir=output_dir,
            output_name=output_name,
            cache_base_dir=cache_dir,
            weights_dir=weights_dir,
        )
        dataset_toml = "/content/musubi_dataset.toml"
        if model_info.get("supports_video"):
            builder.build_dataset_toml(
                dataset_path=dataset_toml,
                resolution=res_list,
                video_folders=datasets,
                batch_size=batch_size,
            )
        else:
            builder.build_dataset_toml(
                dataset_path=dataset_toml,
                resolution=res_list,
                image_folders=datasets,
                batch_size=batch_size,
            )

        vae_path = weights.get("vae", "")
        clip1_path = weights.get("text_encoder1", "")
        clip2_path = weights.get("text_encoder2", None)
        clip_vision = weights.get("clip_vision", None)
        dit_path = weights.get("dit", "")

        cache_latents_cmd = builder.build_cache_latents_args(dataset_toml, vae_path, clip_vision, batch_size) if vae_path else None
        cache_te_cmd = builder.build_cache_text_encoder_args(dataset_toml, clip1_path, clip2_path, batch_size) if clip1_path else None

        train_cmd = builder.build_train_args(
            dataset_config_path=dataset_toml,
            dit_model_path=dit_path,
            learning_rate=learning_rate,
            optimizer_type=optimizer,
            lr_scheduler=lr_scheduler,
            network_dim=network_dim,
            network_alpha=network_alpha,
            max_train_epochs=max_train_epochs,
            save_every_n_epochs=save_every_n_epochs,
            mixed_precision=hw_info["recommended_precision"],
            sample_prompt_file=sample_txt_path,
            sample_every_n_steps=sample_every_n_steps,
        )

        success = run_musubi_pipeline(
            cache_latents_cmd=cache_latents_cmd,
            cache_text_encoder_cmd=cache_te_cmd,
            train_cmd=train_cmd,
        )

    elif engine_type == "toolkit":
        builder = ToolkitConfigBuilder(
            model_name=model_name,
            output_dir=output_dir,
            output_name=output_name,
        )
        yaml_path = "/content/toolkit_config.yaml"
        builder.build_yaml_config(
            save_yaml_path=yaml_path,
            dataset_folders=datasets,
            steps=max_train_epochs * 250, # Ước lượng steps
            save_every=save_every_n_epochs * 250,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lr_scheduler=lr_scheduler,
            linear_dim=network_dim,
            linear_alpha=network_alpha,
            sample_prompts=[final_sample_prompt],
            sample_every=sample_every_n_steps,
            sample_resolution=res_list,
            wandb_api_key=wandb_key,
        )
        success = run_toolkit_pipeline(yaml_path)

    # 7. Tự động chuyển đổi định dạng ComfyUI nếu cần
    if success and convert_to_comfy:
        try:
            auto_convert_checkpoints(output_dir, model_name=model_name)
        except Exception as e:
            print(f"⚠️ Chuyển đổi định dạng LoRA bổ sung: {e}")

    return success
