# 🎨 TranningLoras: Universal & Optimized LoRA Training Suite

> **Bộ công cụ huấn luyện LoRA Đa Nền Tảng Chuyên Sâu trên Google Colab & Linux GPU (Hỗ trợ GPU T4 16GB, L4 24GB, A100 40GB/80GB, V100)**  
> Tích hợp 3 Engine hàng đầu thế giới:
> - **Kohya `sd-scripts`**: SDXL (Pony V6, Illustrious-XL, Animagine XL), SD 1.5, SD 3.5.
> - **Kohya `musubi-tuner`**: MMDiT & Video (Wan 2.1 / 2.2, FLUX.2 Klein, Qwen-Image / Edit, Z-Image Turbo, Krea2).
> - **Ostris `ai-toolkit`**: FLUX.1 (dev, schnell, kontext), SDXL, DoRA/LoRA.

---

## ⚡ Mở Nhanh Trên Google Colab (One-Click Launch)

| Tên Notebook | Mô tả & Hỗ trợ Mô hình | Mở trên Colab |
| :--- | :--- | :---: |
| **01. Universal Image LoRA Trainer** | Huấn luyện LoRA Hình Ảnh đa năng: **Pony V6 XL, Illustrious-XL, Animagine XL, SDXL, SD 1.5, SD 3.5, FLUX.1, FLUX.2 Klein, Qwen-Image / Edit, Z-Image, Krea2, Custom Checkpoints** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenducvuongg/TranningLoras/blob/main/notebooks/01_Universal_Image_LoRA_Trainer.ipynb) |
| **02. Universal Video LoRA Trainer** | Huấn luyện LoRA Video Text-to-Video & Image-to-Video: **Wan 2.1 (14B, 720P, 480P, 1.3B) & Wan 2.2 (14B T2V / I2V)** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenducvuongg/TranningLoras/blob/main/notebooks/02_Universal_Video_LoRA_Trainer.ipynb) |
| **03. Dataset Captioning Tools** | Bộ công cụ gán nhãn AI tự động chuyên sâu: **Gemini 3.7 / 3.6 / 3.5 Flash & Pro, JoyCaption, Florence-2, OpenAI GPT-4o** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenducvuongg/TranningLoras/blob/main/notebooks/03_Dataset_Captioning_Tools.ipynb) |
| **04. Toolkit WebUI Trainer** | Giao diện đồ họa WebUI trực quan của AI-Toolkit qua Colab Port Proxy | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenducvuongg/TranningLoras/blob/main/notebooks/04_Toolkit_WebUI_Trainer.ipynb) |

---

## 🌟 1. Cấu Trúc Thư Mục Chuẩn Hóa Trên Google Drive

Khi chạy Bước 1, hệ thống sẽ **tự động quét kiểm tra** và khởi tạo cây thư mục chuẩn hóa tại `/content/drive/MyDrive/TranningLorasData`.  
> 🛡️ **An toàn tuyệt đối**: Hệ thống quét kiểm tra nếu thư mục đã có sẵn sẽ **bảo lưu nguyên vẹn 100% dữ liệu**, không bao giờ tạo đè hay làm mất dữ liệu của bạn.

```text
/content/drive/MyDrive/TranningLorasData/
├── config/                          # Lưu trữ cấu hình hệ thống & API Key
│   ├── api_vault.json               # API Key Vault tự động lưu (Gemini, HF, WandB, CivitAI...)
│   └── accelerate_config.yaml       # Cấu hình Accelerate tối ưu cho GPU
├── models/                          # KHO LƯU TRỮ MODEL VĨNH VIỄN (0 giây tải lại)
│   ├── dit/                         # Trọng số DiT (Klein, Kontext, Wan, Qwen, Z-Image, Krea2...)
│   ├── sdxl/                        # Trọng số SDXL (Pony, Illustrious, Animagine, Base 1.0...)
│   ├── sd15/                        # Trọng số SD 1.5 (v1-5, Realistic Vision...)
│   ├── vae/                         # Trọng số VAE (flux_vae, qwen_vae, wan_vae, sdxl_vae...)
│   └── text_encoders/               # CLIP-L, T5-XXL, Qwen-VL, Wan-T5...
├── datasets/                        # Thư mục dữ liệu huấn luyện
│   ├── train_data/                  # Chứa ảnh / video mục tiêu (Target Data)
│   │   └── 20_my_character/         # VD: 20 ảnh nhân vật kèm file caption .txt
│   └── control_data/                # Chứa ảnh đầu vào đối chiếu cho Paired LoRA (Skin/Retouch/Upscale)
│       └── paired_skin_raw/         # VD: Ảnh da mụn/mờ có tên file khớp 1-1 với train_data
├── outputs/                         # Nơi xuất file LoRA hoàn thiện (.safetensors)
│   ├── sample_images/               # Ảnh/video test sinh ra qua các step
│   └── ComfyUI_Ready/               # LoRA đã tự động convert sang định dạng ComfyUI
└── engines_cache/                   # Môi trường cache wheels & packages
```

---

## 🚀 2. Hướng Dẫn Chi Tiết Huấn Luyện Từng Dạng LoRA

### A. LoRA Anime / Nhân Vật Pony & SDXL (Anime, Manga, Fanart)
- **Mô hình khuyến nghị**: `Pony-Diffusion-V6-XL` hoặc `Illustrious-XL-v0.1`
- **Engine**: Kohya `sd-scripts`
- **Chuẩn bị dữ liệu**:
  - 20 - 50 ảnh chất lượng cao vào `TranningLorasData/datasets/train_data/20_my_character`.
  - Tự động chuẩn hóa tên tệp và tạo file `.txt` đồng bộ.
- **Tham số khuyến nghị**:
  | Tham số | Giá trị khuyến nghị | Ý nghĩa |
  | :--- | :--- | :--- |
  | **Resolution** | `1024,1024` | Độ phân giải chuẩn cho SDXL/Pony |
  | **Network_Dim (Rank)** | `32` (hoặc `64`) | Dung lượng học nét vẽ & chi tiết nhân vật |
  | **Network_Alpha** | `16` (hoặc `32`) | Alpha = Dim / 2 giúp hội tụ ổn định |
  | **Learning_Rate** | `1e-4` | Tốc độ học chuẩn cho AdamW8bit |
  | **Max_Train_Epochs** | `10` | Tương đương ~1500 - 2500 steps |

---

### B. LoRA Nhân Vật / Chân Dung FLUX (Realistic Portrait & Body)
- **Mô hình khuyến nghị**: `FLUX.1-dev` hoặc `Krea2-Raw`
- **Engine**: Ostris `ai-toolkit` hoặc Kohya `musubi-tuner`
- **Chuẩn bị dữ liệu**:
  - Đặt 15 - 30 ảnh vào `TranningLorasData/datasets/train_data/20_my_character`.
  - Chạy AI Captioning (Gemini 3.7 Flash) với chế độ `Character_Outfit`.
- **Tham số khuyến nghị**:
  | Tham số | Giá trị khuyến nghị | Ý nghĩa |
  | :--- | :--- | :--- |
  | **Resolution** | `1024,1024` | Độ nét cao |
  | **Network_Dim / Alpha** | `32 / 16` (hoặc `16 / 16`) | |
  | **Learning_Rate** | `1e-4` | |
  | **Max_Train_Epochs** | `8 - 10` | |

---

### C. LoRA Xử lý Da / Retouch / Phục hồi / Upscale (Paired Control LoRA)
- **Mô hình khuyến nghị**: `FLUX.1-Kontext-dev` hoặc `Qwen-Image-Edit`
- **Engine**: Kohya `musubi-tuner`
- **Chuẩn bị dữ liệu (Ghép cặp 1-1)**:
  - `control_data`: Chứa ảnh thô / da có khuyết điểm / ảnh nén mờ (`0001.jpg`, `0002.jpg`).
  - `train_data`: Chứa ảnh sau khi đã retouch đẹp hoàn hảo / ảnh 4K nét căng (`0001.jpg`, `0002.jpg`).
- **Tham số khuyến nghị**:
  | Tham số | Giá trị khuyến nghị |
  | :--- | :--- |
  | **Network_Dim / Alpha** | `32 / 16` |
  | **Learning_Rate** | `1e-4` |
  | **Max_Train_Epochs** | `10 - 12` |

---

### D. LoRA Video Động (Video Motion & Physics LoRA)
- **Mô hình khuyến nghị**: `Wan2.1-T2V-14B` (Text-to-Video) hoặc `Wan2.1-I2V-14B-720P` (Image-to-Video), `Wan2.2-T2V-14B`
- **Engine**: Kohya `musubi-tuner`
- **Chuẩn bị dữ liệu**:
  - 10 - 30 video clips ngắn chất lượng cao (`.mp4`), cắt đoạn 3 - 5 giây.
  - Đặt vào thư mục `TranningLorasData/datasets/train_data/my_video_dataset`.
- **Tham số khuyến nghị**:
  | Tham số | Giá trị khuyến nghị |
  | :--- | :--- |
  | **Resolution** | `720,1280` (16:9) hoặc `1280,720` (9:16) |
  | **Target_Frames** | `25` |
  | **Frame_Stride** | `1` |
  | **Network_Dim / Alpha** | `32 / 16` |
  | **Max_Train_Epochs** | `15 - 20` |

---

## 📊 3. Bảng Tổng Hợp Tham Số Vàng (Cheat Sheet)

| Tác vụ Huấn luyện | Mô hình Tối ưu | Engine | Dim / Alpha | Learning Rate | Steps Khuyến nghị | VRAM GPU T4/L4 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pony Anime / Art** | `Pony-Diffusion-V6-XL` | sdscripts | `32 / 16` | `1e-4` | 1500 - 2500 | ~11.5 GB (T4 mượt) |
| **Illustrious Anime** | `Illustrious-XL-v0.1` | sdscripts | `32 / 16` | `1e-4` | 1500 - 2500 | ~11.5 GB (T4 mượt) |
| **SD 1.5 Cổ điển** | `v1-5-pruned-emaonly` | sdscripts | `32 / 16` | `1.5e-4` | 1000 - 1800 | ~7.5 GB |
| **SD 3.5 Large** | `SD3.5-Large` | sdscripts | `32 / 16` | `1e-4` | 1500 - 2000 | ~15.0 GB |
| **Nhân vật FLUX.1** | `FLUX.1-dev` | AI-Toolkit | `16 / 16` | `1e-4` | 1500 - 2000 | ~16.0 GB |
| **Xử lý Da / Retouch** | `FLUX.1-Kontext` | Musubi | `32 / 16` | `1e-4` | 2000 - 2500 | ~18.0 GB (L4/A100) |
| **Klein Thế hệ Mới** | `FLUX.2-klein-9B` | Musubi | `32 / 16` | `1.5e-4` | 1200 - 1800 | ~18.5 GB |
| **Tốc độ Siêu nhanh** | `Z-Image-Turbo` | Musubi | `16 / 8` | `2e-4` | 800 - 1200 | ~14.0 GB (T4 mượt) |
| **Video Text-to-Video** | `Wan2.1-T2V-14B` | Musubi | `32 / 16` | `1e-4` | 2500 - 3500 | ~21.0 GB |
| **Video Image-to-Video**| `Wan2.1-I2V-720P`| Musubi | `32 / 16` | `1e-4` | 2500 - 3500 | ~22.5 GB |

---

## 🔐 4. Quản Lý Khóa Bảo Mật (API Key Vault) & Tải Tốc Độ Cao

1. **API Key Vault Tự Động**:
   - Khi bạn nhập API Key bất kỳ (Gemini, Hugging Face, WandB, CivitAI, OpenAI), hệ thống tự động lưu vào `/content/drive/MyDrive/TranningLorasData/config/api_vault.json`.
   - Các lần chạy sau chỉ cần để trống ô API Key, hệ thống sẽ **tự động lấy key mặc định đã lưu**.
2. **Kho Trọng Số Vĩnh Viễn (Persistent Model Storage)**:
   - Toàn bộ Model Base, VAE và Text Encoder khi tải lần đầu sẽ được lưu trực tiếp vào Google Drive.
   - Các lần chạy sau: Quét phát hiện file đã có sẵn $\rightarrow$ **Bỏ qua bước tải 100% (0 giây chờ)**.
3. **Cơ Chế Phục Hồi Token Thông Minh (Anonymous Recovery)**:
   - Tự động gỡ token và chuyển sang mirror công khai nếu gặp lỗi `401 Unauthorized`.
   - Tự động nhúng CivitAI API token vào URL tải của CivitAI.
