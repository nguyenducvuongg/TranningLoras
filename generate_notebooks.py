"""
Universal Colab Notebook Generator
Tự động tạo ra 4 file Jupyter Notebook hoàn chỉnh, trực quan, tối ưu cho Google Colab:
1. 01_Universal_Image_LoRA_Trainer.ipynb (SDXL, Pony, Illustrious, SD1.5, SD3.5, FLUX.1, FLUX.2, Qwen, Z-Image, Krea, Custom)
2. 02_Universal_Video_LoRA_Trainer.ipynb (Wan 2.1 & Wan 2.2 Video LoRA)
3. 03_Dataset_Captioning_Tools.ipynb (AI Captioning Studio)
4. 04_Toolkit_WebUI_Trainer.ipynb (AI-Toolkit GUI Server)
"""

import json
import os

os.makedirs("notebooks", exist_ok=True)

GITHUB_REPO_URL = "https://github.com/nguyenducvuongg/TranningLoras.git"


def create_cell(cell_type: str, source: str) -> dict:
    lines = source.splitlines()
    src_lines = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines else [])
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": src_lines,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def save_notebook(filepath: str, cells: list) -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "gpuType": "T4",
            },
            "language_info": {
                "name": "python",
            },
        },
        "cells": cells,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"✅ Đã tạo: {filepath}")


# ==============================================================================
# 1. 01_Universal_Image_LoRA_Trainer.ipynb
# ==============================================================================
cells_image = [
    create_cell("markdown", f"""# 🎨 Universal Image LoRA Trainer (Google Colab)
Bộ công cụ huấn luyện LoRA Hình Ảnh đa năng, tối ưu hóa bộ nhớ VRAM cho:
- **SDXL Family**: SDXL Base 1.0, Pony Diffusion V6 XL, Illustrious-XL, Animagine XL, DreamShaper XL
- **SD 1.5 Family**: v1-5-pruned-emaonly, Realistic Vision v5.1, DreamShaper 8
- **SD 3.5 Family**: SD3.5-Large, SD3.5-Large-Turbo, SD3.5-Medium
- **FLUX.1 Family**: FLUX.1-dev, FLUX.1-schnell, FLUX.1-Kontext-dev (Skin/Retouch/Edit)
- **FLUX.2 / Klein**: FLUX.2-klein-base-9B, FLUX.2-klein-base-4B
- **Qwen-Image**: Qwen-Image, Qwen-Image-Edit (2509, 2511)
- **Z-Image & Krea**: Z-Image Turbo / Base, Krea2-Raw, Sana-1.6B
- **Custom Models**: Hỗ trợ tải checkpoint tùy chỉnh trực tiếp từ CivitAI hoặc HuggingFace!

---
### 📚 Hướng Dẫn Chuẩn Bị Dữ Liệu Cho Từng Dạng LoRA:
1. **LoRA Nhân vật / Concept / Phong cách (Standard LoRA)**:
   - Đặt toàn bộ ảnh và file `.txt` caption vào một thư mục (VD: `/content/drive/MyDrive/TranningLorasData/datasets/train_data/20_my_character`).
   - Có thể đặt tên thư mục theo cú pháp `{{repeats}}_{{tên}}` để tự nhận số lần lặp.
2. **LoRA Xử lý Da / Retouch / Phục hồi / Upscale (Paired Control LoRA)**:
   - Chuẩn bị 2 thư mục:
     - `Control_Folder`: Chứa ảnh đầu vào (ảnh da mụn/thô, ảnh mờ/nén).
     - `Train_Folders`: Chứa ảnh kết quả chất lượng cao (ảnh da đã chỉnh đẹp, ảnh gốc 4K siêu nét).
     - **Lưu ý**: Ảnh ở 2 thư mục phải có **tên tệp giống nhau 1-1** (VD: `0001.jpg`, `0002.jpg`).
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
!apt-get install -y -qq aria2 ffmpeg
!pip install -q toml pyyaml python-dotenv bitsandbytes optimum-quanto google-genai openai accelerate safetensors huggingface_hub tqdm pillow ipywidgets voluptuous imagesize einops ftfy regex sentencepiece protobuf scipy wandb lion-pytorch prodigyopt albumentations av opencv-python-headless diffusers imageio imageio-ffmpeg kornia open_clip_torch timm

# Clone / Cập nhật repo chính thức
if not os.path.exists('/content/TranningLoras'):
    !git clone {GITHUB_REPO_URL} /content/TranningLoras
else:
    !git -C /content/TranningLoras pull

%cd /content/TranningLoras
!pip install -q -e .

from lora_trainer.core.hardware import detect_hardware_environment, setup_cuda_environment
from lora_trainer.core.key_vault import display_key_vault_dashboard
from lora_trainer.storage.drive_manager import setup_storage_structure

# Khởi tạo cây thư mục chuẩn trên Google Drive (Quét an toàn, bảo lưu 100% dữ liệu có sẵn)
storage_dirs = setup_storage_structure()
setup_cuda_environment()

hw_info = detect_hardware_environment()

print("\\n" + "="*60)
print(f"🚀 GPU: {{hw_info['gpu_name']}} | VRAM: {{hw_info['vram_gb']}} GB ({{hw_info['device_tier']}})")
print(f"⚡ Khuyến nghị: Precision = {{hw_info['recommended_precision']}} | FP8 = {{hw_info['recommended_fp8']}} | Batch Size = {{hw_info['recommended_batch_size']}}")
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

from lora_trainer.core.key_vault import save_api_key, display_key_vault_dashboard
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
Caption_Engine = "Gemini-3.7-Flash" # @param ["None", "Gemini-3.7-Flash", "Gemini-3.6-Flash", "Gemini-3.5-Flash", "Gemini-3.5-Flash-Lite", "Gemini-3.1-Pro", "Gemini-3-Pro", "Florence-2", "JoyCaption", "OpenAI-GPT4o"]
# @markdown 🎯 **Chế độ Prompt chuyên biệt theo mục đích LoRA:**
Task_Mode = "General" # @param ["General", "Skin_Portrait", "Upscale_Restoration", "Art_Style", "Character_Outfit"]
Caption_Length = "Medium" # @param ["Short", "Medium", "Long"]

# @markdown 🔑 **API Key (Tự động ghi nhớ)**: Để trống nếu muốn dùng Key đã lưu trước đó trong Vault.
API_Key = "" # @param {type:'string'}
Custom_Trigger_Word = "" # @param {type:'string'}
Add_Folder_Name = False # @param {type:'boolean'}
Overwrite_Existing_Captions = False # @param {type:'boolean'}

import os
from lora_trainer.dataset.cleaner import clean_directory
from lora_trainer.dataset.renamer import batch_standardize_datasets
from lora_trainer.captioning.gemini import batch_caption_gemini
from lora_trainer.captioning.florence import batch_caption_florence
from lora_trainer.captioning.joycaption import batch_caption_joy
from lora_trainer.captioning.openai_gpt import batch_caption_openai
from lora_trainer.dataset.tagger import process_dir_tags

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
- **LoRA Pony / Anime (SDXL)**: Chọn `Pony-Diffusion-V6-XL` hoặc `Illustrious-XL-v0.1`, đặt LR = 1e-4, Dim = 32, Alpha = 16.
- **LoRA Xử lý Da / Portrait (FLUX)**: Chọn `FLUX.1-Kontext-dev` hoặc `Qwen-Image-Edit`, nhập `Control_Folder`, đặt LR = 1e-4, Dim = 32, Alpha = 16.
- **LoRA Phong cách (Art Style)**: Chọn `FLUX.2-klein-base-9B` hoặc `Z-Image-Turbo`, đặt LR = 1.5e-4, Dim = 32.
- **Custom Model**: Nhập URL từ CivitAI hoặc HuggingFace vào ô `Custom_Model_URL`.
"""),
    create_cell("code", """# @title 🛠️ 3. Thiết lập Tham số & Bắt đầu Huấn luyện
# @markdown 📂 **Thư mục Dữ liệu**:
Train_Folders = "/content/drive/MyDrive/TranningLorasData/datasets/train_data/my_character" # @param {type:'string'}
Control_Folder = "" # @param {type:'string'}

Model_Type = "Pony-Diffusion-V6-XL" # @param ["Pony-Diffusion-V6-XL", "Illustrious-XL-v0.1", "Animagine-XL-3.1", "SDXL-Base-1.0", "DreamShaper-XL", "v1-5-pruned-emaonly", "Realistic-Vision-v5.1", "DreamShaper-8", "SD3.5-Large", "SD3.5-Large-Turbo", "SD3.5-Medium", "FLUX.1-dev", "FLUX.1-schnell", "FLUX.1-Kontext-dev", "FLUX.2-klein-base-9B", "FLUX.2-klein-base-4B", "Qwen-Image", "Qwen-Image-Edit", "Qwen-Image-Edit-2509", "Qwen-Image-Edit-2511", "Z-Image-Turbo", "Z-Image-Base", "Z-Image-De-Turbo", "Krea2-Raw", "Sana-1.6B", "Custom-SDXL", "Custom-SD15", "Custom-FLUX"]
Custom_Model_URL = "" # @param {type:'string'}

Output_Directory = "/content/drive/MyDrive/TranningLorasData/outputs" # @param {type:'string'}
LoRA_Name = "my_lora_model" # @param {type:'string'}

Resolution = "1024,1024" # @param {type:'string'}
Batch_Size = 1 # @param {type:'integer'}
Learning_Rate = 1e-4 # @param {type:'number'}
Optimizer = "adamw8bit" # @param ["adamw8bit", "adamw", "lion8bit", "prodigy"]
LR_Scheduler = "cosine_with_restarts" # @param ["cosine_with_restarts", "constant", "linear"]
Network_Dim = 32 # @param {type:'integer'}
Network_Alpha = 16 # @param {type:'integer'}

Max_Train_Epochs = 10 # @param {type:'integer'}
Save_Every_N_Epochs = 1 # @param {type:'integer'}
Sample_Every_N_Steps = 200 # @param {type:'integer'}
Sample_Prompt = "" # @param {type:'string'}

# @markdown 📊 **Tùy chọn Bổ sung (Tự động ghi nhớ vào Vault)**:
# @markdown 💡 *Để trống các ô dưới nếu bạn muốn dùng Token/Key đã lưu từ trước hoặc Mirror công khai.*
HF_Token = "" # @param {type:'string'}
CivitAI_API_Key = "" # @param {type:'string'}
WandB_API_Key = "" # @param {type:'string'}
Auto_Disconnect = False # @param {type:'boolean'}

from lora_trainer.engines.unified_trainer import run_unified_training
from lora_trainer.utils.colab_env import auto_disconnect

success = run_unified_training(
    model_name=Model_Type,
    train_folders=Train_Folders,
    output_dir=Output_Directory,
    output_name=LoRA_Name,
    control_folder=Control_Folder if Control_Folder else None,
    resolution=Resolution,
    batch_size=Batch_Size,
    learning_rate=Learning_Rate,
    optimizer=Optimizer,
    lr_scheduler=LR_Scheduler,
    network_dim=Network_Dim,
    network_alpha=Network_Alpha,
    max_train_epochs=Max_Train_Epochs,
    save_every_n_epochs=Save_Every_N_Epochs,
    sample_every_n_steps=Sample_Every_N_Steps,
    sample_prompt=Sample_Prompt,
    hf_token=HF_Token or None,
    civitai_key=CivitAI_API_Key or None,
    wandb_key=WandB_API_Key or None,
    custom_model_url=Custom_Model_URL or None,
)

if success:
    print("\\n🎉 CHÚC MỪNG! HUẤN LUYỆN LORA ĐÃ HOÀN TẤT THÀNH CÔNG VÀ XUẤT RA GOOGLE DRIVE!")
    if Auto_Disconnect:
        auto_disconnect(force=True)
"""),
]

save_notebook("notebooks/01_Universal_Image_LoRA_Trainer.ipynb", cells_image)


# ==============================================================================
# 2. 02_Universal_Video_LoRA_Trainer.ipynb
# ==============================================================================
cells_video = [
    create_cell("markdown", f"""# 🎬 Universal Video LoRA Trainer (Wan 2.1 & Wan 2.2)
Bộ công cụ huấn luyện LoRA Video Đa Năng (Text-to-Video & Image-to-Video) cho:
- **Wan 2.1**: Wan2.1-T2V-14B, Wan2.1-I2V-14B-720P, Wan2.1-I2V-14B-480P, Wan2.1-T2V-1.3B
- **Wan 2.2**: Wan2.2-T2V-14B, Wan2.2-I2V-14B
"""),
    create_cell("markdown", "### ☕ Bước 1: Khởi tạo Môi trường & GPU"),
    create_cell("code", f"""# @title ⚙️ 1. Cài đặt Môi trường
import os
from google.colab import drive
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

!apt-get install -y -qq aria2 ffmpeg
!pip install -q toml pyyaml python-dotenv bitsandbytes optimum-quanto google-genai openai accelerate safetensors huggingface_hub tqdm pillow av opencv-python-headless diffusers open_clip_torch timm

if not os.path.exists('/content/TranningLoras'):
    !git clone {GITHUB_REPO_URL} /content/TranningLoras
else:
    !git -C /content/TranningLoras pull

%cd /content/TranningLoras
!pip install -q -e .

from lora_trainer.core.hardware import detect_hardware_environment, setup_cuda_environment
from lora_trainer.storage.drive_manager import setup_storage_structure

setup_storage_structure()
setup_cuda_environment()
hw = detect_hardware_environment()
print(f"🚀 GPU: {{hw['gpu_name']}} | VRAM: {{hw['vram_gb']}} GB")
"""),
    create_cell("markdown", "### 🎬 Bước 2: Cấu hình Video Dataset & Huấn luyện"),
    create_cell("code", """# @title 🚀 2. Thiết lập Tham số Video & Bắt đầu Huấn luyện
Video_Folder = "/content/drive/MyDrive/TranningLorasData/datasets/train_data/my_video_dataset" # @param {type:'string'}
Model_Type = "Wan2.1-T2V-14B" # @param ["Wan2.1-T2V-14B", "Wan2.1-I2V-14B-720P", "Wan2.1-I2V-14B-480P", "Wan2.1-T2V-1.3B", "Wan2.2-T2V-14B", "Wan2.2-I2V-14B"]

Output_Directory = "/content/drive/MyDrive/TranningLorasData/outputs" # @param {type:'string'}
LoRA_Name = "wan_video_lora" # @param {type:'string'}

Resolution = "720,1280" # @param {type:'string'}
Target_Frames = 25 # @param {type:'integer'}
Frame_Stride = 1 # @param {type:'integer'}
Learning_Rate = 1e-4 # @param {type:'number'}
Network_Dim = 32 # @param {type:'integer'}
Network_Alpha = 16 # @param {type:'integer'}
Max_Train_Epochs = 15 # @param {type:'integer'}
Save_Every_N_Epochs = 1 # @param {type:'integer'}
Sample_Every_N_Steps = 200 # @param {type:'integer'}
Sample_Prompt = "" # @param {type:'string'}
Auto_Disconnect = False # @param {type:'boolean'}

from lora_trainer.engines.unified_trainer import run_unified_training
from lora_trainer.utils.colab_env import auto_disconnect

success = run_unified_training(
    model_name=Model_Type,
    train_folders=Video_Folder,
    output_dir=Output_Directory,
    output_name=LoRA_Name,
    resolution=Resolution,
    learning_rate=Learning_Rate,
    network_dim=Network_Dim,
    network_alpha=Network_Alpha,
    max_train_epochs=Max_Train_Epochs,
    save_every_n_epochs=Save_Every_N_Epochs,
    sample_every_n_steps=Sample_Every_N_Steps,
    sample_prompt=Sample_Prompt,
)

if success and Auto_Disconnect:
    auto_disconnect(force=True)
"""),
]

save_notebook("notebooks/02_Universal_Video_LoRA_Trainer.ipynb", cells_video)


# ==============================================================================
# 3. 03_Dataset_Captioning_Tools.ipynb
# ==============================================================================
cells_caption = [
    create_cell("markdown", f"""# 🏷️ Dataset AI Captioning Studio
Bộ công cụ gán nhãn chuyên sâu tự động với **Gemini 3.7 / 3.6 / 3.5 Flash & Pro, JoyCaption, Florence-2, OpenAI GPT-4o**.
"""),
    create_cell("markdown", "### ☕ Bước 1: Khởi tạo"),
    create_cell("code", f"""# @title ⚙️ 1. Cài đặt Môi trường
import os
from google.colab import drive
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

!pip install -q google-genai openai pillow tqdm transformers bitsandbytes accelerate

if not os.path.exists('/content/TranningLoras'):
    !git clone {GITHUB_REPO_URL} /content/TranningLoras
else:
    !git -C /content/TranningLoras pull

%cd /content/TranningLoras
!pip install -q -e .

from lora_trainer.core.key_vault import display_key_vault_dashboard
display_key_vault_dashboard()
"""),
    create_cell("markdown", "### 🏷️ Bước 2: Gán nhãn AI Hàng loạt"),
    create_cell("code", """# @title 🤖 2. Chọn Thư mục & Bắt đầu Gán nhãn AI
Dataset_Folder = "/content/drive/MyDrive/TranningLorasData/datasets/train_data/my_character" # @param {type:'string'}
Caption_Engine = "Gemini-3.7-Flash" # @param ["Gemini-3.7-Flash", "Gemini-3.6-Flash", "Gemini-3.5-Flash", "Gemini-3.1-Pro", "Florence-2", "JoyCaption", "OpenAI-GPT4o"]
Task_Mode = "General" # @param ["General", "Skin_Portrait", "Upscale_Restoration", "Art_Style", "Character_Outfit"]
Caption_Length = "Medium" # @param ["Short", "Medium", "Long"]

API_Key = "" # @param {type:'string'}
Trigger_Word = "" # @param {type:'string'}
Overwrite = False # @param {type:'boolean'}

from lora_trainer.captioning.gemini import batch_caption_gemini
from lora_trainer.captioning.florence import batch_caption_florence
from lora_trainer.captioning.joycaption import batch_caption_joy
from lora_trainer.captioning.openai_gpt import batch_caption_openai

if Caption_Engine.startswith("Gemini"):
    batch_caption_gemini(Dataset_Folder, api_key=API_Key, model_alias=Caption_Engine, task_mode=Task_Mode, caption_length=Caption_Length, trigger_word=Trigger_Word or None, overwrite=Overwrite)
elif Caption_Engine == "Florence-2":
    batch_caption_florence(Dataset_Folder, task_mode=Task_Mode, trigger_word=Trigger_Word or None, overwrite=Overwrite)
elif Caption_Engine == "JoyCaption":
    batch_caption_joy(Dataset_Folder, task_mode=Task_Mode, trigger_word=Trigger_Word or None, overwrite=Overwrite)
elif Caption_Engine == "OpenAI-GPT4o":
    batch_caption_openai(Dataset_Folder, api_key=API_Key, task_mode=Task_Mode, caption_length=Caption_Length, trigger_word=Trigger_Word or None, overwrite=Overwrite)
"""),
]

save_notebook("notebooks/03_Dataset_Captioning_Tools.ipynb", cells_caption)


# ==============================================================================
# 4. 04_Toolkit_WebUI_Trainer.ipynb
# ==============================================================================
cells_webui = [
    create_cell("markdown", f"""# 🖥️ Ostris AI-Toolkit WebUI Server
Khởi chạy giao diện đồ họa WebUI của AI-Toolkit trực tiếp trên Google Colab qua Port Proxy.
"""),
    create_cell("code", f"""# @title 🚀 1. Cài đặt & Khởi chạy WebUI
import os
from google.colab import drive
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

!apt-get install -y -qq aria2
!pip install -q pyyaml accelerate safetensors huggingface_hub gradio

if not os.path.exists('/content/ai-toolkit'):
    !git clone --recurse-submodules https://github.com/ostris/ai-toolkit.git /content/ai-toolkit

%cd /content/ai-toolkit
!pip install -q -r requirements.txt
!python webui.py --share
"""),
]

save_notebook("notebooks/04_Toolkit_WebUI_Trainer.ipynb", cells_webui)
print("🎉 ĐÃ HOÀN TẤT TẠO TOÀN BỘ NOTEBOOKS MỚI!")
