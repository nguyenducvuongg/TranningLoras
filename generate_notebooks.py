import json
import os

os.makedirs("notebooks", exist_ok=True)

GITHUB_REPO_URL = "https://github.com/nguyenducvuongg/TranningLoras.git"

def create_cell(cell_type, source):
    if isinstance(source, list):
        src_lines = [s + "\n" for s in source[:-1]] + [source[-1]] if source else []
    else:
        lines = source.splitlines()
        src_lines = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])
    
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": src_lines
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell

def save_notebook(filepath, cells):
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "gpuType": "T4"
            },
            "language_info": {
                "name": "python"
            }
        },
        "cells": cells
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"✅ Đã tạo: {filepath}")

# ==============================================================================
# 1. 01_Universal_Image_LoRA_Trainer.ipynb
# ==============================================================================
cells_image = [
    create_cell("markdown", f"""# 🎨 Universal Image LoRA Trainer (Google Colab)
Bộ công cụ huấn luyện LoRA Hình Ảnh đa năng, tối ưu hóa bộ nhớ VRAM cho **FLUX.1, FLUX.2 Klein, Qwen-Image, Qwen-Image-Edit, Z-Image Turbo và Krea2**.

---
### 📚 Hướng Dẫn Chuẩn Bị Dữ Liệu Cho Từng Dạng LoRA:
1. **LoRA Nhân vật / Concept / Phong cách (Standard LoRA)**:
   - Đặt toàn bộ ảnh và file `.txt` caption vào một thư mục (VD: `/content/drive/MyDrive/TranningLorasData/20_model_girl`).
   - Có thể đặt tên thư mục theo cú pháp `{{repeats}}_{{tên}}` để tự nhận số lần lặp.
2. **LoRA Xử lý Da / Retouch / Phục hồi / Upscale (Paired Control LoRA)**:
   - Chuẩn bị 2 thư mục:
     - `Control_Folder`: Chứa ảnh đầu vào (ảnh da mụn/thô, ảnh mờ/nén).
     - `Train_Folders`: Chứa ảnh kết quả chất lượng cao (ảnh da đã chỉnh đẹp, ảnh gốc 4K siêu nét).
     - **Lưu ý**: Ảnh ở 2 thư mục phải có **tên tệp giống nhau 1-1** (VD: `001.jpg`, `002.jpg`).
"""),

    create_cell("markdown", "### ☕ Bước 1: Khởi tạo Môi trường & Nhận diện GPU"),
    create_cell("code", f"""# @title ⚙️ 1. Cài đặt Thư viện & Kiểm tra GPU & Key Vault
# @markdown Nhấn nút Play để cài đặt môi trường và kiểm tra các API Key đã lưu trên Google Drive:

import os
import sys

# Mount Google Drive
from google.colab import drive
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# Cài đặt các gói phụ thuộc & công cụ tải đa luồng aria2
!apt-get install -y -qq aria2
!pip install -q toml pyyaml python-dotenv bitsandbytes optimum-quanto google-genai openai accelerate safetensors huggingface_hub tqdm pillow ipywidgets voluptuous imagesize einops ftfy regex sentencepiece protobuf scipy wandb lion-pytorch prodigyopt albumentations av opencv-python-headless diffusers imageio imageio-ffmpeg kornia open_clip_torch timm

# Clone / Cập nhật repo chính thức
if not os.path.exists('/content/TranningLoras'):
    !git clone {GITHUB_REPO_URL} /content/TranningLoras
else:
    !git -C /content/TranningLoras pull

%cd /content/TranningLoras
!pip install -q -e .

from lora_trainer.engine.hardware import detect_hardware_environment, setup_cuda_environment
from lora_trainer.caption.key_manager import display_key_vault_dashboard
from lora_trainer.engine.model_storage import setup_storage_structure
from lora_trainer.engine.environment_setup import initialize_training_environment

# Khởi tạo cây thư mục chuẩn trên Google Drive (Quét an toàn, bảo lưu 100% dữ liệu có sẵn)
storage_dirs = setup_storage_structure()
env_info = initialize_training_environment()

hw_info = detect_hardware_environment()

print("\\n" + "="*60)
print(f"🚀 GPU: {{hw_info['gpu_name']}} | VRAM: {{hw_info['vram_gb']}} GB ({{hw_info['device_tier']}})")
print(f"⚡ Khuyến nghị: FP8 = {{hw_info['recommended_fp8']}} | Batch Size = {{hw_info['recommended_batch_size']}} | Res = {{hw_info['recommended_resolution']}}")
print("="*60 + "\\n")

# Hiển thị bảng điều khiển API Key Vault đã lưu từ các phiên trước
display_key_vault_dashboard()
"""),

    create_cell("markdown", "### 🔐 (Tùy chọn) Quản lý & Thêm API Key vào Vault"),
    create_cell("code", """# @title 🔐 Quản lý API Key Vault (Thêm / Đổi Key)
# @markdown Dùng form này nếu bạn muốn lưu trước API Key mới vào Google Drive:
Platform = "gemini" # @param ["gemini", "huggingface", "wandb", "openai", "civitai"]
New_API_Key = "" # @param {type:'string'}
Key_Label = "" # @param {type:'string'}

from lora_trainer.caption.key_manager import save_api_key, display_key_vault_dashboard
if New_API_Key.strip():
    save_api_key(Platform, New_API_Key, label=Key_Label or None, set_default=True)
display_key_vault_dashboard()
"""),

    create_cell("markdown", "### 📂 Bước 2: Chuẩn bị Dữ liệu & AI Captioning Chuyên Sâu"),
    create_cell("code", """# @title 📂 2. Cấu hình Dữ liệu & AI Captioning
# @markdown Nhập đường dẫn thư mục ảnh trên Google Drive:

Train_Folders = "/content/drive/MyDrive/TranningLorasData/datasets/train_data/my_character" # @param {type:'string'}
# @markdown 💡 `Control_Folder chỉ dùng khi train LoRA dạng Edit / Inpainting / Xử lý da / Upscale (như FLUX Kontext, Qwen Edit)`
Control_Folder = "" # @param {type:'string'}
Clean_Data = True # @param {type:'boolean'}

# @markdown 🔢 **Chuẩn hóa Tên File (Tự động đổi tên ảnh & caption .txt theo thứ tự 0001, 0002...)**
Standardize_Dataset_Names = True # @param {type:'boolean'}
Filename_Prefix = "" # @param {type:'string'}

# @markdown 🤖 **Tùy chọn AI Captioning**
Caption_Engine = "Gemini-3.6-Flash" # @param ["None", "Gemini-3.6-Flash", "Gemini-3.7-Flash", "Gemini-3.5-Flash", "Gemini-3.5-Flash-Lite", "Gemini-3.1-Pro", "Gemini-3-Pro", "Florence-2", "JoyCaption", "OpenAI-GPT4o"]
# @markdown 🎯 **Chế độ Prompt chuyên biệt theo mục đích LoRA:**
Task_Mode = "General" # @param ["General", "Skin_Portrait", "Upscale_Restoration", "Art_Style", "Character_Outfit"]
Caption_Length = "Medium" # @param ["Short", "Medium", "Long"]

# @markdown 🔑 **API Key (Tự động ghi nhớ)**: Để trống nếu muốn dùng Key đã lưu trước đó trong Vault.
API_Key = "" # @param {type:'string'}
Custom_Trigger_Word = "" # @param {type:'string'}
Add_Folder_Name = False # @param {type:'boolean'}
Overwrite_Existing_Captions = False # @param {type:'boolean'}

from lora_trainer.data.cleaner import clean_directory
from lora_trainer.data.renamer import batch_standardize_datasets
from lora_trainer.caption.gemini_captioner import batch_caption_gemini
from lora_trainer.caption.florence_captioner import batch_caption_florence
from lora_trainer.caption.joy_captioner import batch_caption_joy
from lora_trainer.caption.openai_captioner import batch_caption_openai
from lora_trainer.data.tag_processor import process_dir_tags

# 1. Dọn dẹp & Chuẩn hóa tên tệp đồng bộ (Ảnh + Caption + Control)
if Standardize_Dataset_Names:
    batch_standardize_datasets(
        train_folders=Train_Folders,
        control_folders=Control_Folder if Control_Folder else None,
        prefix=Filename_Prefix,
        digits=4,
        auto_create_txt=True,
    )

for t_dir in [d.strip() for d in Train_Folders.split(",") if d.strip()]:
    if Clean_Data and not Standardize_Dataset_Names:
        clean_directory(t_dir)

    if Caption_Engine.startswith("Gemini"):
        batch_caption_gemini(
            t_dir,
            api_key=API_Key,
            model_alias=Caption_Engine,
            task_mode=Task_Mode,
            caption_length=Caption_Length,
            trigger_word=Custom_Trigger_Word or None,
            overwrite=Overwrite_Existing_Captions,
        )
    elif Caption_Engine == "Florence-2":
        batch_caption_florence(t_dir, task_mode=Task_Mode, trigger_word=Custom_Trigger_Word or None, overwrite=Overwrite_Existing_Captions)
    elif Caption_Engine == "JoyCaption":
        batch_caption_joy(t_dir, task_mode=Task_Mode, trigger_word=Custom_Trigger_Word or None, overwrite=Overwrite_Existing_Captions)
    elif Caption_Engine == "OpenAI-GPT4o":
        batch_caption_openai(t_dir, api_key=API_Key, task_mode=Task_Mode, trigger_word=Custom_Trigger_Word or None, overwrite=Overwrite_Existing_Captions)

    if Custom_Trigger_Word or Add_Folder_Name:
        tag = Custom_Trigger_Word if Custom_Trigger_Word else os.path.basename(t_dir)
        process_dir_tags(t_dir, tag)

if Control_Folder and os.path.exists(Control_Folder):
    if Clean_Data and not Standardize_Dataset_Names:
        clean_directory(Control_Folder)
    print(f"✅ Thư mục Control đối chiếu: {Control_Folder}")
"""),

    create_cell("markdown", """### 🚀 Bước 3: Cấu hình Model & Khởi chạy Huấn luyện
💡 **Gợi ý thiết lập cho các bài toán đặc biệt:**
- **LoRA Xử lý Da / Portrait**: Chọn `FLUX.1-Kontext-dev` hoặc `Qwen-Image-Edit`, nhập `Control_Folder`, đặt LR = 1e-4, Dim = 32, Alpha = 16.
- **LoRA Upscale / Tăng nét**: Chọn `FLUX.1-Kontext-dev`, đặt Dim = 16, Alpha = 16, LR = 1.5e-4.
- **LoRA Phong cách (Art Style)**: Chọn `FLUX.2-klein-base-9B` hoặc `Z-Image-Turbo`, không nhập Control_Folder, đặt LR = 1e-4, Dim = 32.
"""),
    create_cell("code", """# @title 🛠️ 3. Thiết lập Tham số & Bắt đầu Huấn luyện
# @markdown 📂 **Thư mục Dữ liệu**:
Train_Folders = "/content/drive/MyDrive/TranningLorasData/datasets/train_data/my_character" # @param {type:'string'}
Control_Folder = "" # @param {type:'string'}

Model_Type = "Krea2-Raw" # @param ["Krea2-Raw", "FLUX.1-dev", "FLUX.1-schnell", "FLUX.1-Kontext-dev", "FLUX.2-klein-base-9B", "FLUX.2-klein-base-4B", "Qwen-Image", "Qwen-Image-Edit", "Qwen-Image-Edit-2509", "Qwen-Image-Edit-2511", "Z-Image-Turbo", "Z-Image-Base", "Z-Image-De-Turbo"]

Output_Directory = "/content/drive/MyDrive/TranningLorasData/outputs" # @param {type:'string'}
LoRA_Name = "my_character_lora" # @param {type:'string'}

Resolution = "1024,1024" # @param {type:'string'}
Batch_Size = 1 # @param {type:'integer'}
Learning_Rate = 1e-4 # @param {type:'number'}
Optimizer = "adamw8bit" # @param ["adamw8bit", "adamw", "lion8bit", "prodigy"]
LR_Scheduler = "constant" # @param ["constant", "cosine", "linear"]
Network_Dim = 32 # @param {type:'integer'}
Network_Alpha = 16 # @param {type:'integer'}

Max_Train_Epochs = 8 # @param {type:'integer'}
Save_Every_N_Epochs = 1 # @param {type:'integer'}
Sample_Every_N_Steps = 200 # @param {type:'integer'}
Sample_Prompt = "" # @param {type:'string'}

# @markdown 📊 **Tùy chọn Bổ sung (Tự động ghi nhớ vào Vault)**:
# @markdown 💡 *Để trống các ô dưới nếu bạn muốn dùng Token/Key đã lưu từ trước hoặc Mirror công khai.*
HF_Token = "" # @param {type:'string'}
WandB_API_Key = "" # @param {type:'string'}
Auto_Disconnect = False # @param {type:'boolean'}

import os
from lora_trainer.config.model_registry import get_preferred_engine
from lora_trainer.config.musubi_config import MusubiConfigBuilder
from lora_trainer.config.toolkit_config import ToolkitConfigBuilder
from lora_trainer.engine.downloader import download_model_suite
from lora_trainer.engine.musubi_runner import run_musubi_pipeline
from lora_trainer.engine.toolkit_runner import run_toolkit_pipeline
from lora_trainer.data.dataset_builder import build_dataset_list
from lora_trainer.utils.sampler import get_random_sample_prompt
from lora_trainer.utils.converter import auto_convert_checkpoints
from lora_trainer.utils.colab_utils import auto_disconnect

res_list = [int(x.strip()) for x in Resolution.split(",") if x.strip()]
if len(res_list) == 1:
    res_list = [res_list[0], res_list[0]]

datasets = build_dataset_list(Train_Folders, Control_Folder)
engine_type = get_preferred_engine(Model_Type)
print(f"🎯 Mô hình: {Model_Type} | Engine tối ưu: {engine_type.upper()}")

# Tải trước các trọng số cần thiết (Tự động ưu tiên kho lưu trữ vĩnh viễn trên Google Drive)
weights = download_model_suite(
    Model_Type,
    weights_dir="/content/models",
    hf_token=HF_Token,
    base_drive_dir="/content/drive/MyDrive/TranningLorasData",
)

if engine_type == "musubi":
    builder = MusubiConfigBuilder(
        model_name=Model_Type,
        output_dir=Output_Directory,
        output_name=LoRA_Name,
        cache_base_dir="/content/cache",
        weights_dir="/content/models",
    )
    
    dataset_toml = "/content/dataset.toml"
    builder.build_dataset_toml(
        dataset_path=dataset_toml,
        resolution=res_list,
        image_folders=datasets,
    )

    vae_path = weights.get("vae", "")
    clip1_path = weights.get("text_encoder1", "")
    clip2_path = weights.get("text_encoder2", None)
    clip_vision = weights.get("clip_vision", None)
    dit_path = weights.get("dit", "")

    cache_latents_cmd = builder.build_cache_latents_args(dataset_toml, vae_path, clip_vision) if vae_path else None
    cache_te_cmd = builder.build_cache_text_encoder_args(dataset_toml, clip1_path, clip2_path) if clip1_path else None

    sample_txt_path = "/content/sample_prompt.txt"
    if Sample_Prompt == "":
        p, img, ctrl = get_random_sample_prompt(datasets[0]["path"], datasets[0].get("control_path"))
        Sample_Prompt = p
    with open(sample_txt_path, "w", encoding="utf-8") as f:
        f.write(f"{Sample_Prompt} --w {res_list[0]} --h {res_list[1]}\\n")

    train_cmd = builder.build_train_args(
        dataset_config_path=dataset_toml,
        dit_model_path=dit_path,
        learning_rate=Learning_Rate,
        optimizer_type=Optimizer,
        lr_scheduler=LR_Scheduler,
        network_dim=Network_Dim,
        network_alpha=Network_Alpha,
        max_train_epochs=Max_Train_Epochs,
        save_every_n_epochs=Save_Every_N_Epochs,
        sample_prompt_file=sample_txt_path if Sample_Every_N_Steps > 0 else None,
        sample_every_n_steps=Sample_Every_N_Steps,
        wandb_api_key=WandB_API_Key if WandB_API_Key else None,
    )

    run_musubi_pipeline(
        musubi_dir="/content/musubi-tuner",
        cache_latents_cmd=cache_latents_cmd,
        cache_text_encoder_cmd=cache_te_cmd,
        train_cmd=train_cmd,
    )

else:
    builder = ToolkitConfigBuilder(
        model_name=Model_Type,
        output_dir=Output_Directory,
        output_name=LoRA_Name,
    )
    yaml_path = "/content/toolkit_config.yaml"
    builder.build_yaml_config(
        save_yaml_path=yaml_path,
        dataset_folders=datasets,
        steps=Max_Train_Epochs * 200,
        save_every=Save_Every_N_Epochs * 200,
        batch_size=Batch_Size,
        learning_rate=Learning_Rate,
        linear_dim=Network_Dim,
        linear_alpha=Network_Alpha,
        sample_prompts=[Sample_Prompt] if Sample_Prompt else None,
        sample_every=Sample_Every_N_Steps,
        sample_resolution=res_list,
        wandb_api_key=WandB_API_Key if WandB_API_Key else None,
    )
    run_toolkit_pipeline(config_yaml_path=yaml_path, toolkit_dir="/content/ai-toolkit")

# ⚡ TỰ ĐỘNG NHẬN DIỆN VÀ CHUYỂN ĐỔI SANG COMFYUI NẾU MÔ HÌNH YÊU CẦU (Ví dụ Z-Image)
auto_convert_checkpoints(Output_Directory, Model_Type)

if Auto_Disconnect:
    auto_disconnect(delay_seconds=120, enabled=True)
"""),

    create_cell("markdown", """### 🛠️ Bước 4: Công Cụ Chuyển Đổi Thủ Công LoRA Sang ComfyUI (Tùy Chọn)
💡 **Lưu ý**: Hệ thống ở Bước 3 đã **tự động phát hiện và chuyển đổi** các file LoRA (như Z-Image) sang định dạng ComfyUI trong thư mục `ComfyUI_Ready`.
Nếu bạn muốn tự chuyển đổi thủ công một file bất kỳ:
- **`Input_LoRA`**: Đường dẫn file `.safetensors` gốc sau khi train (ví dụ: `/content/drive/MyDrive/LoRA_Outputs/my_awesome_lora-000010.safetensors`).
- **`Output_LoRA`**: Đường dẫn file `.safetensors` mới sẵn sàng cho ComfyUI (ví dụ: `/content/drive/MyDrive/LoRA_Outputs/comfy_my_awesome_lora-000010.safetensors`).
"""),
    create_cell("code", """# @title 🔄 Convert Z-LoRA to ComfyUI (Manual Tool)
Input_LoRA = "/content/drive/MyDrive/LoRA_Outputs/my_awesome_lora-000010.safetensors" # @param {type:'string'}
Output_LoRA = "/content/drive/MyDrive/LoRA_Outputs/comfy_my_awesome_lora.safetensors" # @param {type:'string'}

from lora_trainer.utils.converter import convert_z_lora_to_comfyui
if Input_LoRA and Output_LoRA and os.path.exists(Input_LoRA):
    convert_z_lora_to_comfyui(Input_LoRA, Output_LoRA)
else:
    print("⚠️ Vui lòng kiểm tra lại đường dẫn file Input_LoRA!")
""")
]

save_notebook("notebooks/01_Universal_Image_LoRA_Trainer.ipynb", cells_image)

# ==============================================================================
# 2. 02_Universal_Video_LoRA_Trainer.ipynb
# ==============================================================================
cells_video = [
    create_cell("markdown", f"""# 🎥 Universal Video LoRA Trainer (Wan 2.1 & Wan 2.2)
Hệ thống huấn luyện LoRA Video chuyên sâu trên Google Colab hỗ trợ **Text-to-Video** và **Image-to-Video** với **Wan 2.1** và **Wan 2.2**.

---
### 📚 Hướng Dẫn Chuẩn Bị Video Dataset:
- Chuẩn bị 10 - 50 video clips ngắn (3 - 10 giây mỗi clip), định dạng `.mp4`.
- Tỉ lệ khuyến nghị: 16:9 (`720,1280`) hoặc 9:16 (`1280,720`).
- Đặt lượng frame mục tiêu (`Target_Frames`): 25, 33, 49 hoặc 81 frames.
"""),

    create_cell("markdown", "### ☕ Bước 1: Khởi tạo Môi trường & Kiểm tra GPU & Key Vault"),
    create_cell("code", f"""# @title ⚙️ 1. Cài đặt Môi trường
import os
from google.colab import drive
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

!apt-get install -y -qq aria2
!pip install -q toml pyyaml python-dotenv bitsandbytes optimum-quanto google-genai openai accelerate safetensors huggingface_hub tqdm pillow av opencv-python-headless voluptuous imagesize einops ftfy regex sentencepiece protobuf scipy wandb lion-pytorch prodigyopt albumentations diffusers imageio imageio-ffmpeg kornia open_clip_torch timm

if not os.path.exists('/content/TranningLoras'):
    !git clone {GITHUB_REPO_URL} /content/TranningLoras
else:
    !git -C /content/TranningLoras pull

%cd /content/TranningLoras
!pip install -q -e .

from lora_trainer.engine.hardware import detect_hardware_environment
from lora_trainer.caption.key_manager import display_key_vault_dashboard
from lora_trainer.engine.model_storage import setup_storage_structure
from lora_trainer.engine.environment_setup import initialize_training_environment

# Khởi tạo cây thư mục chuẩn trên Google Drive (Quét an toàn, bảo lưu 100% dữ liệu có sẵn)
storage_dirs = setup_storage_structure()
env_info = initialize_training_environment()

hw = detect_hardware_environment()
print(f"🚀 GPU: {{hw['gpu_name']}} | VRAM: {{hw['vram_gb']}} GB")
display_key_vault_dashboard()
"""),

    create_cell("markdown", "### 📂 Bước 2: Dữ liệu Video, Frame Slicing & AI Captioning"),
    create_cell("code", """# @title 📂 2. Xử lý Dữ liệu Video & Cắt Frame
Video_Folders = "/content/drive/MyDrive/TranningLorasData/datasets/train_data/my_video_data" # @param {type:'string'}

# @markdown 🎞️ **Cấu hình Trích xuất Khung hình (Frame Extraction)**
Frame_Extraction = "chunk" # @param ["chunk", "slide", "uniform", "head", "full"]
Target_Frames = "25" # @param {type:'string'}
Frame_Stride = 1 # @param {type:'integer'}
Frame_Sample = 1 # @param {type:'integer'}
Max_Frames = 33 # @param {type:'integer'}

# @markdown 🤖 **Tự động gán nhãn Video bằng Gemini API (Đọc video trực tiếp)**
Auto_Caption_Video = False # @param {type:'boolean'}
Gemini_Model = "Gemini-3.6-Flash" # @param ["Gemini-3.6-Flash", "Gemini-3.7-Flash", "Gemini-3.5-Flash", "Gemini-3.5-Flash-Lite", "Gemini-3.1-Pro", "Gemini-3-Pro"]
# @markdown 🔑 **Gemini API Key (Tự động ghi nhớ)**: Để trống nếu muốn dùng Key đã lưu trong Vault.
Gemini_API_Key = "" # @param {type:'string'}
Custom_Tag = "" # @param {type:'string'}

from lora_trainer.data.cleaner import clean_directory, get_supported_videos
from lora_trainer.caption.gemini_captioner import batch_caption_gemini
from lora_trainer.data.tag_processor import process_dir_tags

for v_dir in [d.strip() for d in Video_Folders.split(",") if d.strip()]:
    clean_directory(v_dir)
    if Auto_Caption_Video:
        batch_caption_gemini(v_dir, api_key=Gemini_API_Key, model_alias=Gemini_Model, is_video_folder=True)
    if Custom_Tag:
        process_dir_tags(v_dir, Custom_Tag)
    vids = get_supported_videos(v_dir)
    print(f"📹 Thư mục {v_dir}: {len(vids)} video.")
"""),

    create_cell("markdown", "### 🚀 Bước 3: Cấu hình Wan & Bắt đầu Huấn luyện"),
    create_cell("code", """# @title 🛠️ 3. Cấu hình Wan 2.1 / Wan 2.2 & Bắt đầu Huấn luyện
# @markdown 📂 **Thư mục Dữ liệu Video**:
Video_Folders = "/content/drive/MyDrive/TranningLorasData/datasets/train_data/my_video_data" # @param {type:'string'}
Frame_Extraction = "chunk" # @param ["chunk", "slide", "uniform", "head", "full"]
Target_Frames = "25" # @param {type:'string'}
Frame_Stride = 1 # @param {type:'integer'}
Frame_Sample = 1 # @param {type:'integer'}
Max_Frames = 33 # @param {type:'integer'}

Model_Type = "Wan2.2-T2V-14B" # @param ["Wan2.1-T2V-14B", "Wan2.1-I2V-14B-720P", "Wan2.1-I2V-14B-480P", "Wan2.1-T2V-1.3B", "Wan2.2-T2V-14B", "Wan2.2-I2V-14B"]

Output_Directory = "/content/drive/MyDrive/TranningLorasData/outputs" # @param {type:'string'}
LoRA_Name = "my_video_lora" # @param {type:'string'}

Resolution = "720,1280" # @param {type:'string'}
Learning_Rate = 1e-4 # @param {type:'number'}
Num_Repeats = 10 # @param {type:'integer'}
Max_Train_Epochs = 5 # @param {type:'integer'}
Save_Every_N_Epochs = 1 # @param {type:'integer'}

# @markdown 💡 **Timestep Sampling & Wan 2.2 Boundary**
Timestep_Sampling = "shift" # @param ["shift", "sigma", "uniform", "sigmoid", "logsnr"]
Timestep_Boundary = 875 # @param {"type":"slider","min":0,"max":1000,"step":5}
Sample_Prompt = "" # @param {type:'string'}
Sample_Every_N_Steps = 200 # @param {type:'integer'}

# @markdown 📊 **Tùy chọn Bổ sung (Token / WandB)**:
HF_Token = "" # @param {type:'string'}
WandB_API_Key = "" # @param {type:'string'}
Auto_Disconnect = False # @param {type:'boolean'}

import os
from lora_trainer.config.musubi_config import MusubiConfigBuilder
from lora_trainer.engine.downloader import download_model_suite
from lora_trainer.engine.musubi_runner import run_musubi_pipeline
from lora_trainer.utils.colab_utils import auto_disconnect

res = [int(x.strip()) for x in Resolution.split(",")]
tf = [int(x.strip()) for x in Target_Frames.split(",")]

weights = download_model_suite(
    Model_Type,
    weights_dir="/content/models",
    hf_token=HF_Token,
    base_drive_dir="/content/drive/MyDrive/TranningLorasData",
)

builder = MusubiConfigBuilder(
    model_name=Model_Type,
    output_dir=Output_Directory,
    output_name=LoRA_Name,
)

v_list = []
for vd in [d.strip() for d in Video_Folders.split(",") if d.strip()]:
    v_list.append({
        "path": vd,
        "repeats": Num_Repeats,
        "frame_extraction": Frame_Extraction,
        "target_frames": tf,
        "frame_stride": Frame_Stride,
        "frame_sample": Frame_Sample,
        "max_frames": Max_Frames,
    })

dataset_toml = "/content/dataset_video.toml"
builder.build_dataset_toml(
    dataset_path=dataset_toml,
    resolution=res,
    video_folders=v_list,
)

vae_path = weights.get("vae", "")
t5_path = weights.get("text_encoder1", "")
cv_path = weights.get("clip_vision", None)
dit_path = weights.get("dit", "")

cache_latents = builder.build_cache_latents_args(dataset_toml, vae_path, cv_path)
cache_te = builder.build_cache_text_encoder_args(dataset_toml, t5_path)

sample_txt = "/content/prompt_video.txt"
with open(sample_txt, "w", encoding="utf-8") as f:
    f.write(f"{Sample_Prompt} --w {res[0]} --h {res[1]} --f {tf[0]}\\n")

train_cmd = builder.build_train_args(
    dataset_config_path=dataset_toml,
    dit_model_path=dit_path,
    learning_rate=Learning_Rate,
    network_dim=32,
    network_alpha=16,
    max_train_epochs=Max_Train_Epochs,
    save_every_n_epochs=Save_Every_N_Epochs,
    timestep_sampling=Timestep_Sampling,
    timestep_boundary=Timestep_Boundary if "wan22" in Model_Type.lower() else None,
    sample_prompt_file=sample_txt if Sample_Every_N_Steps > 0 else None,
    sample_every_n_steps=Sample_Every_N_Steps,
)

run_musubi_pipeline(
    musubi_dir="/content/musubi-tuner",
    cache_latents_cmd=cache_latents,
    cache_text_encoder_cmd=cache_te,
    train_cmd=train_cmd,
)

if Auto_Disconnect:
    auto_disconnect(delay_seconds=120, enabled=True)
""")
]

save_notebook("notebooks/02_Universal_Video_LoRA_Trainer.ipynb", cells_video)

# ==============================================================================
# 3. 03_Dataset_Captioning_Tools.ipynb
# ==============================================================================
cells_caption = [
    create_cell("markdown", f"""# 📝 AI Dataset Captioning & Tagging Studio
Bộ công cụ chuyên dụng để dọn dẹp, phân tích và gán nhãn tự động cho tập dữ liệu hình ảnh & video sử dụng **Gemini 3.x, Florence-2, JoyCaption, OpenAI**.
"""),

    create_cell("markdown", "### ☕ Bước 1: Cài đặt Môi trường & Key Vault"),
    create_cell("code", f"""# @title ⚙️ 1. Cài đặt Môi trường
import os
from google.colab import drive
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

!pip install -q google-genai openai transformers accelerate pillow tqdm

if not os.path.exists('/content/TranningLoras'):
    !git clone {GITHUB_REPO_URL} /content/TranningLoras
else:
    !git -C /content/TranningLoras pull

%cd /content/TranningLoras
!pip install -q -e .

from lora_trainer.caption.key_manager import display_key_vault_dashboard
print("✅ Đã cài đặt hoàn tất!")
display_key_vault_dashboard()
"""),

    create_cell("markdown", "### 🔐 (Tùy chọn) Quản lý API Key Vault"),
    create_cell("code", """# @title 🔐 Quản lý API Key Vault (Thêm / Đổi Key)
Platform = "gemini" # @param ["gemini", "huggingface", "wandb", "openai", "civitai"]
New_API_Key = "" # @param {type:'string'}
Key_Label = "" # @param {type:'string'}

from lora_trainer.caption.key_manager import save_api_key, display_key_vault_dashboard
if New_API_Key.strip():
    save_api_key(Platform, New_API_Key, label=Key_Label or None, set_default=True)
display_key_vault_dashboard()
"""),

    create_cell("markdown", "### 📂 Bước 2: AI Captioning Studio"),
    create_cell("code", """# @title ✨ 2. Gán nhãn Tự động
Dataset_Folder = "/content/drive/MyDrive/My_Dataset" # @param {type:'string'}
Caption_Engine = "Gemini-3.6-Flash" # @param ["Gemini-3.6-Flash", "Gemini-3.7-Flash", "Gemini-3.5-Flash", "Gemini-3.5-Flash-Lite", "Gemini-3.1-Pro", "Gemini-3-Pro", "Florence-2", "JoyCaption", "OpenAI-GPT4o"]
Task_Mode = "General" # @param ["General", "Skin_Portrait", "Upscale_Restoration", "Art_Style", "Character_Outfit"]
Caption_Length = "Medium" # @param ["Short", "Medium", "Long"]

# @markdown 🔑 **API Key (Tự động ghi nhớ)**: Để trống nếu muốn dùng Key đã lưu trong Vault.
API_Key = "" # @param {type:'string'}
Overwrite_Existing = False # @param {type:'boolean'}
Is_Video_Dataset = False # @param {type:'boolean'}

from lora_trainer.caption.gemini_captioner import batch_caption_gemini
from lora_trainer.caption.florence_captioner import batch_caption_florence
from lora_trainer.caption.joy_captioner import batch_caption_joycaption
from lora_trainer.caption.openai_captioner import batch_caption_openai

if Caption_Engine.startswith("Gemini"):
    batch_caption_gemini(
        Dataset_Folder,
        api_key=API_Key,
        model_alias=Caption_Engine,
        length_preset=Caption_Length,
        task_mode=Task_Mode,
        overwrite=Overwrite_Existing,
        is_video_folder=Is_Video_Dataset,
    )
elif Caption_Engine == "Florence-2":
    batch_caption_florence(Dataset_Folder, task_preset=Caption_Length, overwrite=Overwrite_Existing)
elif Caption_Engine == "JoyCaption":
    batch_caption_joycaption(Dataset_Folder, caption_length=Caption_Length.lower(), overwrite=Overwrite_Existing)
elif Caption_Engine == "OpenAI-GPT4o":
    batch_caption_openai(Dataset_Folder, api_key=API_Key, overwrite=Overwrite_Existing)
"""),

    create_cell("markdown", "### 🏷️ Bước 3: Thao tác Tag Nâng Cao (Thêm/Sửa/Xóa Trigger Words)"),
    create_cell("code", """# @title 🏷️ 3. Quản lý Thẻ Tag & Trigger Word
Dataset_Folder = "/content/drive/MyDrive/My_Dataset" # @param {type:'string'}
Trigger_Word = "" # @param {type:'string'}
Action = "Prepend" # @param ["Prepend", "Append", "Remove", "Add_Folder_Name"]

from lora_trainer.data.tag_processor import process_dir_tags, add_folder_name_tags

if Action == "Prepend":
    process_dir_tags(Dataset_Folder, Trigger_Word, append=False)
elif Action == "Append":
    process_dir_tags(Dataset_Folder, Trigger_Word, append=True)
elif Action == "Remove":
    process_dir_tags(Dataset_Folder, Trigger_Word, remove_tag=True)
elif Action == "Add_Folder_Name":
    add_folder_name_tags(Dataset_Folder)

print("✅ Đã cập nhật xong toàn bộ tag!")
"""),

    create_cell("markdown", """### 🔢 Bước 4: Chuẩn Hóa Tên Tệp Dataset Tự Động (Renamer Tool)
💡 Tự động quét và đổi tên toàn bộ ảnh và file `.txt` caption trong thư mục thành dạng chuẩn (VD: `0001.png`, `0001.txt` hoặc `char_0001.jpg`, `char_0001.txt`), đồng thời đồng bộ thư mục Control nếu có.
"""),
    create_cell("code", """# @title 🔢 4. Chuẩn Hóa & Đổi Tên Tệp Hàng Loạt
Train_Folders = "/content/drive/MyDrive/My_Dataset" # @param {type:'string'}
Control_Folder = "" # @param {type:'string'}
Filename_Prefix = "" # @param {type:'string'}
Number_Of_Digits = 4 # @param {type:'integer'}
Auto_Create_Missing_Captions = True # @param {type:'boolean'}
Default_Caption_Text = "" # @param {type:'string'}

from lora_trainer.data.renamer import batch_standardize_datasets

batch_standardize_datasets(
    train_folders=Train_Folders,
    control_folders=Control_Folder if Control_Folder else None,
    prefix=Filename_Prefix,
    digits=Number_Of_Digits,
    auto_create_txt=Auto_Create_Missing_Captions,
    default_caption=Default_Caption_Text,
)
""")
]

save_notebook("notebooks/03_Dataset_Captioning_Tools.ipynb", cells_caption)

# ==============================================================================
# 4. 04_Toolkit_WebUI_Trainer.ipynb
# ==============================================================================
cells_webui = [
    create_cell("markdown", f"""# 🌐 AI-Toolkit WebUI Launcher (Google Colab Proxy)
Khởi chạy giao diện WebUI trực quan của Ostris AI-Toolkit thông qua **Google Colab Port Proxy** gốc (an toàn, không cần Cloudflare hay tài khoản bên ngoài).
"""),

    create_cell("markdown", "### ☕ Khởi chạy WebUI"),
    create_cell("code", f"""# @title 🚀 Launch AI-Toolkit WebUI
import os
from google.colab import drive
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

if not os.path.exists('/content/TranningLoras'):
    !git clone {GITHUB_REPO_URL} /content/TranningLoras
else:
    !git -C /content/TranningLoras pull

%cd /content/TranningLoras
!pip install -q -e .

if not os.path.exists('/content/ai-toolkit'):
    !git clone --recurse-submodules https://github.com/ostris/ai-toolkit /content/ai-toolkit
else:
    !git -C /content/ai-toolkit pull
    !git -C /content/ai-toolkit submodule update --init --recursive

%cd /content/ai-toolkit
!pip install -q -r requirements.txt
!pip install -q diffusers transformers accelerate safetensors bitsandbytes optimum-quanto torchvision albumentations opencv-python-headless pyyaml toml pillow tqdm scipy wandb tensorboard matplotlib einops imagesize ftfy regex sentencepiece protobuf av imageio imageio-ffmpeg kornia open_clip_torch timm prodigyopt lion-pytorch voluptuous huggingface_hub flatten_json pydantic clean-fid invisible-watermark
!pip install -q -e .
!npm --prefix /content/ai-toolkit/ui install

from lora_trainer.utils.colab_utils import launch_colab_proxy

# Tạo link proxy cho cổng 8675
launch_colab_proxy(8675)

%cd /content/ai-toolkit/ui
!npm run build_and_start
""")
]

save_notebook("notebooks/04_Toolkit_WebUI_Trainer.ipynb", cells_webui)
