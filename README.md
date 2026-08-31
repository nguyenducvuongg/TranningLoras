# 🚀 Universal Colab LoRA Trainer (Image & Video)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Train Image LoRA on Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenducvuongg/TranningLoras/blob/main/notebooks/01_Universal_Image_LoRA_Trainer.ipynb)
[![Train Video LoRA on Colab](https://img.shields.io/badge/Colab-Train%20Video%20LoRA-orange?logo=googlecolab)](https://colab.research.google.com/github/nguyenducvuongg/TranningLoras/blob/main/notebooks/02_Universal_Video_LoRA_Trainer.ipynb)

Bộ công cụ huấn luyện LoRA toàn diện, độc lập và tối ưu hóa hiệu năng cao nhất trên môi trường **Google Colab** (hỗ trợ GPU T4 16GB Free, L4 24GB, A100 40/80GB Pro/Pro+).

Hỗ trợ đầy đủ các kiến trúc Diffusion & Flow-Matching mới nhất cho cả **Hình ảnh** (FLUX 1/2, Qwen-Image, Z-Image Turbo, Krea2) và **Video** (Wan 2.1, Wan 2.2).

---

## 🌟 Tính Năng Nổi Bật

- ⚡ **Dual-Engine Auto Dispatcher**: Tự động chuyển đổi giữa **Kohya Musubi-Tuner** (Pre-caching Latents & Text Encoders giúp tiết kiệm 70% VRAM) và **Ostris AI-Toolkit** (linh hoạt cho FLUX.1).
- 🔄 **Always Up-to-Date**: Luôn tự động cập nhật các commit mới nhất và submodules của Kohya Musubi-Tuner và Ostris AI-Toolkit khi khởi chạy.
- 🎬 **Chuyên Sâu Huấn Luyện Video (Wan 2.1 & Wan 2.2)**: Hỗ trợ 5 thuật toán trích xuất frame (`chunk`, `slide`, `uniform`, `head`, `full`), hỗ trợ Timestep Boundary cho Wan 2.2 dual-subnet.
- 🤖 **Universal AI Captioning Hub**: Tích hợp toàn bộ dòng Google Gemini API (**Gemini 2.5 Pro, 2.5 Flash, 2.5 Flash Lite, 2.0 Flash, 2.0 Pro Exp, 1.5 Pro, 1.5 Flash**) phân tích trực tiếp cả **Hình ảnh** và **Video**, đi kèm các chế độ prompt chuyên sâu cho từng bài toán (Da, Upscale, Phong cách, Nhân vật). Tích hợp Florence-2 và JoyCaption Alpha Two chạy offline miễn phí.
- 🌐 **Google Colab Native Proxy**: Kết nối WebUI trực tiếp qua cổng nội bộ của Colab, an toàn, tốc độ cao, không phụ thuộc Cloudflare hay tài khoản ngoài.
- 🧹 **Tiền Xử Lý Dữ Liệu Thông Minh**: Tự động làm sạch file rác (`.DS_Store`, `._*`, 0-byte), trích xuất số bước/repeats từ tên folder `{steps}_{concept}`, ghép cặp 1-1 thư mục điều kiện (Multi-Control 1/2/3).
- 🔄 **Xuất Định Dạng Tiêu Chuẩn**: Hỗ trợ chuyển đổi sang định dạng ComfyUI native (Z-LoRA to ComfyUI) và Diffusers sang Single Safetensors.

---

## 📂 Hướng Dẫn Chuẩn Bị Thư Mục Dữ Liệu (Dataset Preparation)

Tùy theo loại LoRA muốn huấn luyện, bạn chuẩn bị cấu trúc thư mục trên Google Drive theo các hướng dẫn chuẩn dưới đây:

### 1. Dạng LoRA Tiêu Chuẩn: Nhân vật / Concept / Style
Dành cho việc học một khuôn mặt, nhân vật, trang phục hoặc phong cách cụ thể.
- **Số lượng**: 15 - 50 ảnh chất lượng cao, đa dạng góc chụp và ánh sáng.
- **Cấu trúc thư mục**:
  ```text
  MyDrive/LoRA_Data/
  └── 20_vietnamese_girl/            # Cú pháp: {repeats}_{trigger_word}
      ├── 001.jpg
      ├── 001.txt                    # Caption mô tả
      ├── 002.jpg
      └── 002.txt
  ```

### 2. Dạng LoRA Ghép Cặp (Paired Control LoRA): Xử lý Da / Upscale / Edit / Inpainting
Dành cho mô hình điều kiện (FLUX Kontext, Qwen-Image-Edit, Flux.2 Klein) để học sự biến đổi từ trạng thái **A (Đầu vào)** sang **B (Đầu ra)**.
- **Quy tắc quan trọng**: Thư mục Train và Control phải chứa các ảnh có **tên tệp giống nhau 1-1**.
- **Cấu trúc thư mục**:
  ```text
  MyDrive/LoRA_Data/
  ├── Control_Data/                  # [Ảnh Đầu Vào]
  │   ├── img_001.jpg                # (Ảnh da mụn / ảnh mờ / ảnh chưa sửa)
  │   ├── img_002.jpg
  │   └── img_003.jpg
  │
  └── Train_Data/                    # [Ảnh Kết Quả Chuẩn]
      ├── img_001.jpg                # (Ảnh da đã retouch đẹp / ảnh 4K siêu nét)
      ├── img_001.txt                # (Caption mô tả)
      ├── img_002.jpg
      ├── img_002.txt
      ├── img_003.jpg
      └── img_003.txt
  ```

### 3. Dạng LoRA Video (Wan 2.1 & Wan 2.2)
- **Số lượng**: 10 - 50 video clips ngắn (3 - 10 giây mỗi clip), định dạng `.mp4`.
- **Tỉ lệ**: Chuẩn 16:9 (`720,1280`) hoặc 9:16 (`1280,720`).
- **Cấu trúc thư mục**:
  ```text
  MyDrive/LoRA_Video_Data/
  ├── clip_001.mp4
  ├── clip_001.txt                   # (Tùy chọn nếu dùng Auto Caption Gemini)
  ├── clip_002.mp4
  └── clip_002.txt
  ```

---

## 🎯 Cẩm Nang Huấn Luyện Các Dòng LoRA Chuyên Sâu

### 💆 1. LoRA Xử Lý Da & Retouch Chân Dung Thực Tế (Skin Retouching)
- **Mục tiêu**: Làm mịn da tự nhiên, xóa khuyết điểm, mụn, thâm nhưng **vẫn giữ nguyên lỗ chân lông (pores) và kết cấu da thật**.
- **Kiến trúc đề xuất**: `FLUX.1-Kontext-dev` hoặc `Qwen-Image-Edit`.
- **Dữ liệu**: Bộ ảnh Paired (`Control_Folder` chứa ảnh da thô/mụn/ánh sáng xấu; `Train_Folders` chứa ảnh da đã retouch chuyên nghiệp).
- **Chiến lược Caption**: Sử dụng Gemini với chế độ `Task_Mode = Skin_Portrait` để mô tả kỹ kết cấu da tự nhiên và ánh sáng chân dung.
- **Tham số tối ưu**:
  - `Resolution`: `1024,1024`
  - `Network_Dim`: `32` | `Network_Alpha`: `16`
  - `Learning_Rate`: `1e-4`
  - `Timestep_Sampling`: `shift` hoặc `sigma` (tập trung bước thấp 0 - 300 để tinh chỉnh micro-details).
  - `Epochs`: 8 - 12 epochs.

---

### 🔍 2. LoRA Upscale / Tăng Nét & Phục Hồi Chi Tiết (Super-Resolution)
- **Mục tiêu**: Khôi phục ảnh mờ, khử nhiễu, tăng chi tiết sợi vải, sợi tóc, bề mặt vật liệu.
- **Kiến trúc đề xuất**: `FLUX.1-Kontext-dev` hoặc `Qwen-Image-Edit`.
- **Dữ liệu**: Paired Dataset (`Control_Folder` chứa ảnh downscale/nén mờ; `Train_Folders` chứa ảnh gốc độ phân giải cao 2K/4K).
- **Chiến lược Caption**: Sử dụng Gemini với chế độ `Task_Mode = Upscale_Restoration` tập trung vào độ sắc nét, micro-details.
- **Tham số tối ưu**:
  - `Network_Dim`: `16` | `Network_Alpha`: `16`
  - `Learning_Rate`: `1.5e-4`
  - `Loss`: `mse` hoặc `flowmatch`
  - `Epochs`: 5 - 10 epochs.

---

### 🎨 3. LoRA Phong Cách Nghệ Thuật & Màu Ảnh (Artistic / Photography Style)
- **Mục tiêu**: Học phong cách màu film, cyberpunk, anime, tranh sơn dầu, ánh sáng điện ảnh.
- **Kiến trúc đề xuất**: `FLUX.2-klein-base-9B`, `Z-Image-Turbo`, hoặc `FLUX.1-dev`.
- **Dữ liệu**: 20 - 50 ảnh chuẩn phong cách (không cần Control Folder).
- **Chiến lược Caption**:
  - Sử dụng Gemini với chế độ `Task_Mode = Art_Style`.
  - **Mẹo sống còn**: Chỉ mô tả nội dung vật thể (con người, đồ vật, bối cảnh), **KHÔNG** đưa các từ khóa chỉ phong cách (như "oil painting", "cyberpunk color") vào trong caption. Điều này ép mô hình phải gán toàn bộ đặc trưng phong cách vào **Trigger Word**!
- **Tham số tối ưu**:
  - `Learning_Rate`: `1e-4`
  - `Network_Dim`: `32` | `Network_Alpha`: `16`
  - `Timestep_Sampling`: `shift` hoặc `logsnr` (tập trung bước cao 400 - 1000 để học bố cục và màu sắc tổng thể).
  - `Epochs`: 10 - 15 epochs.

---

### 🎬 4. LoRA Video (Wan 2.1 & Wan 2.2)
- **Kiến trúc**: `Wan2.1-T2V-14B` (T2V), `Wan2.1-I2V-14B-720P` (I2V), `Wan2.2-T2V-14B`, `Wan2.2-I2V-14B`.
- **Chiến lược**:
  - Sử dụng Gemini Video Captioning để phân tích chuyển động camera và hành động.
  - Bật Pre-caching Latents & Text Encoders để giảm tối đa VRAM.
  - Đối với **Wan 2.2**: Thiết lập `Timestep_Boundary = 875` (cho T2V) hoặc `900` (cho I2V) để điều phối giữa 2 subnet high-noise và low-noise.

---

## 💻 Danh Sách Colab Notebooks

| Notebook | Mục đích | Mở Trực Tiếp Trên Colab (1 Click) |
| :--- | :--- | :---: |
| [01_Universal_Image_LoRA_Trainer.ipynb](notebooks/01_Universal_Image_LoRA_Trainer.ipynb) | Huấn luyện LoRA Hình Ảnh (FLUX.1, FLUX.2 Klein, Qwen, Z-Image, Krea) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenducvuongg/TranningLoras/blob/main/notebooks/01_Universal_Image_LoRA_Trainer.ipynb) |
| [02_Universal_Video_LoRA_Trainer.ipynb](notebooks/02_Universal_Video_LoRA_Trainer.ipynb) | Huấn luyện LoRA Video (Wan 2.1 & Wan 2.2 T2V/I2V) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenducvuongg/TranningLoras/blob/main/notebooks/02_Universal_Video_LoRA_Trainer.ipynb) |
| [03_Dataset_Captioning_Tools.ipynb](notebooks/03_Dataset_Captioning_Tools.ipynb) | Studio xử lý dataset, dọn dẹp và AI Captioning chuyên sâu | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenducvuongg/TranningLoras/blob/main/notebooks/03_Dataset_Captioning_Tools.ipynb) |
| [04_Toolkit_WebUI_Trainer.ipynb](notebooks/04_Toolkit_WebUI_Trainer.ipynb) | Khởi chạy AI-Toolkit WebUI qua Colab Port Proxy | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenducvuongg/TranningLoras/blob/main/notebooks/04_Toolkit_WebUI_Trainer.ipynb) |

---

## 🛠️ Kiểm Thử Tự Động (Unit Tests)

```bash
python3 -m unittest tests/test_all.py
```

---

## 📄 Giấy Phép Mã Nguồn (License)

Dự án này được phân phối dưới giấy phép **[MIT License](LICENSE)**. Bạn hoàn toàn tự do sử dụng, chỉnh sửa, phân phối và sử dụng cho mục đích cá nhân hoặc thương mại.

