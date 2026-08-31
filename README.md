# 🚀 Universal Colab LoRA Trainer (Image & Video)

Bộ công cụ huấn luyện LoRA toàn diện, độc lập và tối ưu hóa hiệu năng cao nhất trên môi trường **Google Colab** (hỗ trợ GPU T4 16GB Free, L4 24GB, A100 40/80GB Pro/Pro+).

Hỗ trợ đầy đủ các kiến trúc Diffusion & Flow-Matching mới nhất cho cả **Hình ảnh** (FLUX 1/2, Qwen-Image, Z-Image Turbo, Krea2) và **Video** (Wan 2.1, Wan 2.2).

---

## 🌟 Tính Năng Nổi Bật

- ⚡ **Dual-Engine Auto Dispatcher**: Tự động chuyển đổi giữa **Kohya Musubi-Tuner** (Pre-caching Latents & Text Encoders giúp tiết kiệm 70% VRAM) và **Ostris AI-Toolkit** (linh hoạt cho FLUX.1).
- 🎬 **Chuyên Sâu Huấn Luyện Video (Wan 2.1 & Wan 2.2)**: Hỗ trợ 5 thuật toán trích xuất frame (`chunk`, `slide`, `uniform`, `head`, `full`), hỗ trợ Timestep Boundary cho Wan 2.2 dual-subnet.
- 🤖 **Universal AI Captioning Hub**: Tích hợp Google Gemini 2.5 API (phân tích đồng thời cả **Hình ảnh** và **Video trực tiếp**), Florence-2, JoyCaption Alpha Two, OpenAI Vision và WD14 Tagger.
- 🧹 **Tiền Xử Lý Dữ Liệu Thông Minh**: Tự động làm sạch file rác (`.DS_Store`, `._*`, 0-byte), trích xuất số bước/repeats từ tên folder `{steps}_{concept}`, ghép cặp 1-1 thư mục điều kiện (Multi-Control 1/2/3).
- 🔄 **Xuất Định Dạng Tiêu Chuẩn**: Hỗ trợ chuyển đổi sang định dạng ComfyUI native (Z-LoRA to ComfyUI) và Diffusers sang Single Safetensors.
- 🔌 **Google Drive Sync & Auto-Disconnect**: Lưu checkpoint trực tiếp vào Google Drive và tự động giải phóng Colab Runtime sau khi hoàn thành.

---

## 📂 Cấu Trúc Mã Nguồn

```
TranningLoras/
├── lora_trainer/                      # Gói Python lõi (Core Package)
│   ├── config/                        # Quản lý Model Registry & Cấu hình
│   │   ├── model_registry.py          # Danh mục URL model, VAE, Text Encoder chính thức
│   │   ├── musubi_config.py           # Sinh cấu hình TOML & CLI cho Musubi-Tuner
│   │   └── toolkit_config.py          # Sinh cấu hình YAML cho AI-Toolkit
│   ├── data/                          # Pipeline xử lý dữ liệu đa phương tiện
│   │   ├── cleaner.py                 # Làm sạch dữ liệu, loại bỏ file rác hệ điều hành
│   │   ├── tag_processor.py           # Thêm/sửa/xóa trigger word, thẻ tag, prefix/suffix
│   │   ├── video_processor.py         # Trích xuất khung hình video chuyên sâu
│   │   └── dataset_builder.py         # Quét dataset {steps}_{name}, ghép cặp multi-control
│   ├── caption/                       # Bộ công cụ AI Captioning
│   │   ├── gemini_captioner.py        # Gemini 2.5 API đa phương tiện (Ảnh & Video)
│   │   ├── openai_captioner.py        # OpenAI GPT Vision API
│   │   ├── florence_captioner.py      # Florence-2 Vision Model (Offline)
│   │   ├── joy_captioner.py           # JoyCaption Alpha Two Vision Adapter (Offline)
│   │   └── key_manager.py             # Quản lý API Key an toàn
│   ├── engine/                        # Tầng thực thi huấn luyện
│   │   ├── hardware.py                # Tự động nhận diện GPU (A100, L4, T4) & Cuda Arch
│   │   ├── downloader.py              # Download model siêu tốc qua Aria2c
│   │   ├── musubi_runner.py           # Quản lý Pre-caching và chạy Accelerate
│   │   └── toolkit_runner.py          # Quản lý chạy AI-Toolkit
│   └── utils/                         # Tiện ích mở rộng
│       ├── converter.py               # Chuyển đổi format Z-LoRA sang ComfyUI
│       ├── sampler.py                 # Lấy mẫu xem trước (Preview) ngẫu nhiên
│       └── colab_utils.py             # Mount Google Drive, Auto Disconnect, Cloudflare Tunnel
│
├── notebooks/                         # 4 Interactive Colab Notebooks
│   ├── 01_Universal_Image_LoRA_Trainer.ipynb   # Huấn luyện LoRA Ảnh (Flux, Qwen, Z-Image, Krea)
│   ├── 02_Universal_Video_LoRA_Trainer.ipynb   # Huấn luyện LoRA Video (Wan 2.1, Wan 2.2)
│   ├── 03_Dataset_Captioning_Tools.ipynb       # Công cụ AI Captioning & Xử lý Dataset
│   └── 04_Toolkit_WebUI_Trainer.ipynb         # WebUI qua Cloudflare Tunnel
│
├── tests/                             # Unit tests tự động kiểm thử toàn bộ hệ thống
│   └── test_all.py
├── requirements.txt                   # Phụ thuộc đầy đủ
├── requirements-colab.txt             # Phụ thuộc tối ưu cho Colab
├── setup.py                           # Cài đặt pip package
└── README.md
```

---

## 🚀 Danh Sách Mô Hình Hỗ Trợ

| Nhóm | Mô hình | Chế độ | Engine Tối Ưu |
| :--- | :--- | :--- | :--- |
| **FLUX** | `FLUX.1-dev`, `FLUX.1-schnell` | Image (T2I) | AI-Toolkit |
| **FLUX Kontext** | `FLUX.1-Kontext-dev` | Image Inpainting/Edit | Musubi / Toolkit |
| **FLUX.2 Klein** | `FLUX.2-klein-base-9B`, `FLUX.2-klein-base-4B` | Image (T2I/Edit) | Musubi-Tuner |
| **Qwen-Image** | `Qwen-Image`, `Qwen-Image-Edit` (2509, 2511) | Image (T2I/Edit) | Musubi-Tuner |
| **Z-Image** | `Z-Image-Turbo`, `Z-Image-Base`, `Z-Image-De-Turbo` | Image (T2I Fast) | Musubi-Tuner |
| **Wan 2.1 Video** | `Wan2.1-T2V-14B`, `Wan2.1-I2V-14B-720P/480P`, `Wan2.1-1.3B` | Video (T2V & I2V) | Musubi-Tuner |
| **Wan 2.2 Video** | `Wan2.2-T2V-14B`, `Wan2.2-I2V-14B` (Timestep Boundary) | Video (T2V & I2V) | Musubi-Tuner |
| **Krea & SDXL** | `Krea2-Raw`, `SDXL-1.0` | Image (T2I) | Musubi / Toolkit |

---

## 📖 Hướng Dẫn Sử Dụng Trên Google Colab

### Cách 1: Sử dụng Colab Notebooks trực quan (Khuyến nghị)
1. Mở một trong 4 notebook trong thư mục `notebooks/` trên Google Colab.
2. Thực hiện theo **3 bước đơn giản**:
   - **Bước 1**: Nhấn nút chạy để cài đặt môi trường và kiểm tra GPU.
   - **Bước 2**: Nhập đường dẫn thư mục ảnh/video trên Google Drive và chọn công cụ AI Captioning.
   - **Bước 3**: Chọn Mô hình, điều chỉnh Epochs/Steps, Learning Rate và nhấn **Run Training**.

### Cách 2: Sử dụng qua Python Package
```python
from lora_trainer.config.musubi_config import MusubiConfigBuilder
from lora_trainer.engine.musubi_runner import run_musubi_pipeline
from lora_trainer.data.dataset_builder import build_dataset_list

# 1. Quét dữ liệu
datasets = build_dataset_list("/content/drive/MyDrive/MyConcept")

# 2. Sinh cấu hình
builder = MusubiConfigBuilder(
    model_name="FLUX.2-klein-base-9B",
    output_dir="/content/drive/MyDrive/Outputs",
    output_name="my_lora",
)
dataset_toml = builder.build_dataset_toml(
    dataset_path="/content/dataset.toml",
    resolution=[1024, 1024],
    image_folders=datasets,
)

# 3. Huấn luyện
# ...
```

---

## 🛠️ Chạy Kiểm Thử Tự Động (Unit Tests)

```bash
python3 -m unittest tests/test_all.py
```
