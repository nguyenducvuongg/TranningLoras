"""
Universal Unified LoRA Trainer Facade
Lớp điều phối trung tâm tự động phân giải mô hình, tải trọng số, sinh cấu hình và kích hoạt Engine tối ưu nhất
(Kohya sd-scripts, Kohya Musubi-Tuner, hoặc Ostris AI-Toolkit).
"""

import os
import math
from typing import Dict, Any, List, Optional, Union
from ..core.model_registry import get_model_info, get_preferred_engine
from ..core.hardware import detect_hardware_environment, setup_cuda_environment
from ..core.key_vault import save_api_key
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
from ..ui.dashboard import create_dashboard


def run_unified_training(
    model_name: str,
    train_folders: str,
    output_dir: str = "/content/drive/MyDrive/TranningLorasData/outputs",
    output_name: str = "my_lora",
    control_folder: Optional[str] = None,
    task_type: str = "general",
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
    noise_offset: Optional[float] = None,
    min_snr_gamma: Optional[int] = None,
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
    Hàm thực thi huấn luyện Universal LoRA Hình Ảnh Chuyên Sâu:
    Hỗ trợ 6 dạng bài toán: Face (Likeness), Character, Skin/Retouch, Art Style, Upscale, Product Commercials.
    Tích hợp bộ lọc chống da nhựa/AI look (Noise Offset, Min-SNR Gamma, No-Half VAE) để đạt độ trung thực cao nhất.
    """
    setup_cuda_environment()
    hw_info = detect_hardware_environment()

    # 0. Tự động lưu Token vào Vault nếu có cung cấp
    if hf_token and hf_token.strip():
        save_api_key("huggingface", hf_token.strip())
    if civitai_key and civitai_key.strip():
        save_api_key("civitai", civitai_key.strip())
    if wandb_key and wandb_key.strip():
        save_api_key("wandb", wandb_key.strip())

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

    # 3. Phân giải Engine tối ưu & thiết lập siêu tham số chống nhựa
    model_info = get_model_info(model_name, custom_download_url=custom_model_url)
    engine_type = engine_override if engine_override else get_preferred_engine(model_name)
    arch = model_info.get("arch", "sdxl")

    # Tự động gán Noise Offset & Min-SNR Gamma tối ưu theo từng bài toán để triệt tiêu da nhựa
    active_noise_offset = noise_offset
    if active_noise_offset is None:
        if task_type.lower() in ["face", "skin", "product"]:
            active_noise_offset = 0.06  # Tạo độ sâu, chống da nhờn/bóng búp bê
        else:
            active_noise_offset = 0.05

    active_min_snr = min_snr_gamma if min_snr_gamma is not None else 5

    print("\n" + "=" * 60)
    print(f"🎯 UNIVERSAL IMAGE LORA TRAINER (CHỐNG NHỰA & 100% LIKENESS)")
    print(f"📌 Mô hình: {model_name} (Kiến trúc: {arch.upper()})")
    print(f"🎯 Bài toán: {task_type.upper()} | Noise Offset: {active_noise_offset} | Min-SNR: {active_min_snr}")
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

    # 5. Xử lý Sample Prompt an toàn
    sample_txt_path = "/content/sample_prompt.txt"
    final_sample_prompt = sample_prompt.strip()
    if not final_sample_prompt:
        p, _, _ = get_random_sample_prompt(datasets[0]["path"], datasets[0].get("control_path"))
        final_sample_prompt = p
    clean_sample_prompt = final_sample_prompt.replace("\n", " ").strip()
    with open(sample_txt_path, "w", encoding="utf-8") as f:
        f.write(f"{clean_sample_prompt} --w {res_list[0]} --h {res_list[1]}\n")

    # 6. Khởi tạo Live Training Dashboard
    total_effective_images = sum(d.get("image_count", 0) * d.get("repeats", 1) for d in datasets)
    if total_effective_images == 0:
        total_effective_images = 20 * 10
    steps_per_epoch = max(1, math.ceil(total_effective_images / batch_size))
    approx_total_steps = max(100, max_train_epochs * steps_per_epoch)

    dashboard = create_dashboard(
        model_name=model_name,
        engine_name=engine_type,
        task_type=task_type,
        lora_name=output_name,
        total_steps=approx_total_steps,
        total_epochs=max_train_epochs,
        output_dir=output_dir,
        anti_plastic_info={
            "noise_offset": active_noise_offset,
            "min_snr": active_min_snr,
            "no_half_vae": True,
            "sdpa": True,
        },
    )
    dashboard.render(force=True)

    success = False

    # 7. Điều hướng Engine thực thi
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

        # Trên GPU T4 16GB, tự động cache text encoder để chống OOM
        should_cache_te = (hw_info["vram_gb"] <= 16.5)

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
            noise_offset=active_noise_offset,
            min_snr_gamma=active_min_snr,
            no_half_vae=True,
            cache_text_encoder_outputs=should_cache_te,
        )
        dashboard.skip_stage(1, reason="SDXL VAE On-the-fly")
        if should_cache_te:
            dashboard.set_stage(2, "running", "Pre-cache Text Encoder...")
        else:
            dashboard.skip_stage(2, reason="Direct TE Embeddings")
        dashboard.set_stage(3, "running", "Đang tối ưu tham số LoRA...")

        success = run_sdscripts_pipeline(train_cmd, dashboard=dashboard)
        dashboard.finish(success=success)

    elif engine_type == "musubi":
        builder = MusubiConfigBuilder(
            model_name=model_name,
            output_dir=output_dir,
            output_name=output_name,
            cache_base_dir=cache_dir,
            weights_dir=weights_dir,
        )
        dataset_toml = "/content/musubi_dataset.toml"
        builder.build_dataset_toml(
            dataset_path=dataset_toml,
            resolution=res_list,
            image_folders=datasets,
            batch_size=batch_size,
        )

        vae_path = weights.get("vae", "")
        clip1_path = weights.get("text_encoder1", "")
        clip2_path = weights.get("text_encoder2", None)
        dit_path = weights.get("dit", "")

        cache_latents_cmd = builder.build_cache_latents_args(dataset_toml, vae_path, batch_size=batch_size) if vae_path else None
        cache_te_cmd = builder.build_cache_text_encoder_outputs_args(dataset_toml, clip1_path, clip2_path, batch_size) if clip1_path else None

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
            dashboard=dashboard,
        )

    elif engine_type == "toolkit":
        # Tính toán số bước (steps) chính xác theo dataset
        dynamic_steps = approx_total_steps
        dynamic_save_every = max(50, save_every_n_epochs * steps_per_epoch)

        builder = ToolkitConfigBuilder(
            model_name=model_name,
            output_dir=output_dir,
            output_name=output_name,
            model_path=weights.get("dit"),
        )
        yaml_path = "/content/toolkit_config.yaml"
        builder.build_yaml_config(
            save_yaml_path=yaml_path,
            dataset_folders=datasets,
            steps=dynamic_steps,
            save_every=dynamic_save_every,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lr_scheduler=lr_scheduler,
            linear_dim=network_dim,
            linear_alpha=network_alpha,
            sample_prompts=[clean_sample_prompt],
            sample_every=sample_every_n_steps,
            sample_resolution=res_list,
            wandb_api_key=wandb_key,
            low_vram=(hw_info["vram_gb"] < 24.0),
            quantize=True,
            model_path=weights.get("dit"),
        )
        success = run_toolkit_pipeline(yaml_path, dashboard=dashboard)

    # 7. Tự động xuất LoRA ComfyUI Ready nếu cần
    if success and convert_to_comfy:
        try:
            auto_convert_checkpoints(output_dir, model_name=model_name)
        except Exception as e:
            print(f"⚠️ Xuất định dạng LoRA bổ sung: {e}")

    return success
