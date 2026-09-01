# 🎨 TranningLoras: Universal & Optimized LoRA Training Suite

> **Bộ công cụ huấn luyện LoRA Đa Năng Chuyên Sâu trên Google Colab & Linux GPU (Hỗ trợ GPU L4 24GB, A100 40GB/80GB, T4 16GB, V100)**  
> Tích hợp 2 Engine hàng đầu thế giới: **Kohya Musubi-Tuner** (MMDiT, Video, Wan 2.1/2.2, Qwen, Z-Image, Krea2) và **Ostris AI-Toolkit** (FLUX.1, SDXL).

---

## ⚡ Mở Nhanh Trên Google Colab (One-Click Launch)

| Tên Notebook | Mô tả & Hỗ trợ Mô hình | Mở trên Colab |
| :--- | :--- | :---: |
| **01. Universal Image LoRA Trainer** | Huấn luyện LoRA Hình Ảnh đa năng: **FLUX.1, FLUX.2 Klein, Kontext, Krea2-Raw, Qwen-Image, Z-Image Turbo** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenducvuongg/TranningLoras/blob/main/notebooks/01_Universal_Image_LoRA_Trainer.ipynb) |
| **02. Universal Video LoRA Trainer** | Huấn luyện LoRA Video Text-to-Video & Image-to-Video: **Wan 2.1 (14B, 720P, 480P) & Wan 2.2** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenducvuongg/TranningLoras/blob/main/notebooks/02_Universal_Video_LoRA_Trainer.ipynb) |
| **03. Dataset Captioning Tools** | Bộ công cụ gán nhãn AI tự động chuyên sâu: **Gemini 3.6/3.7 Flash, JoyCaption, Florence-2, GPT-4o** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenducvuongg/TranningLoras/blob/main/notebooks/03_Dataset_Captioning_Tools.ipynb) |
| **04. Toolkit WebUI Trainer** | Giao diện đồ họa WebUI trực quan của AI-Toolkit qua Colab Port Proxy | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenducvuongg/TranningLoras/blob/main/notebooks/04_Toolkit_WebUI_Trainer.ipynb) |

---

## 🌟 1. Cấu Trúc Thư Mục Chuẩn Hóa Trên Google Drive

Khi chạy Bước 1, hệ thống sẽ **tự động quét kiểm tra** và khởi tạo cây thư mục chuẩn hóa tại `/content/drive/MyDrive/TranningLorasData`.  
> 🛡️ **An toàn tuyệt đối**: Hệ thống quét kiểm tra nếu thư mục đã có sẵn sẽ **bảo lưu nguyên vẹn 100% dữ liệu**, không bao giờ tạo đè hay làm mất dữ liệu của bạn.

```text
/content/drive/MyDrive/TranningLorasData/
├── config/                          # Lưu trữ cấu hình hệ thống & API Key
│   ├── api_vault.json               # API Key Vault tự động lưu (Gemini, HF, WandB...)
│   └── accelerate_config.yaml       # Cấu hình Accelerate tối ưu cho GPU
├── models/                          # KHO LƯU TRỮ MODEL VĨNH VIỄN (0 giây tải lại)
│   ├── dit/                         # Trọng số Base DiT (Krea2, Klein, Kontext, Wan, Qwen...)
│   ├── vae/                         # Trọng số VAE (flux_vae, qwen_vae, wan_vae...)
│   └── text_encoders/               # CLIP-L, T5-XXL, Qwen-VL...
├── datasets/                        # Thư mục dữ liệu huấn luyện
│   ├── train_data/                  # Chứa ảnh / video mục tiêu (Target Data)
│   │   └── 20_my_character/         # VD: 20 ảnh nhân vật kèm file caption .txt
│   └── control_data/                # Chứa ảnh đầu vào đối chiếu cho Paired LoRA
│       └── paired_skin_raw/         # VD: Ảnh da mụn/mờ có tên file khớp 1-1 với train_data
├── outputs/                         # Nơi xuất file LoRA hoàn thiện (.safetensors)
│   ├── sample_images/               # Ảnh/video test sinh ra qua các step
│   └── ComfyUI_Ready/               # LoRA đã tự động convert sang định dạng ComfyUI
└── engines_cache/                   # Môi trường cache wheels & packages
```

---

## 🚀 2. Hướng Dẫn Chi Tiết Huấn Luyện Từng Dạng LoRA

### A. LoRA Nhân Vật / Chân Dung / Body (Character & Portrait LoRA)
- **Mô hình khuyến nghị**: `Krea2-Raw` hoặc `FLUX.1-dev`
- **Engine**: Kohya Musubi-Tuner (cho Krea2) hoặc AI-Toolkit (cho FLUX.1-dev)
- **Chuẩn bị dữ liệu**:
  - Đặt toàn bộ 15 - 30 ảnh chất lượng cao vào `TranningLorasData/datasets/train_data/20_my_character`.
  - Chạy AI Captioning (Gemini 3.6 Flash / Joy Caption) với chế độ `Character_Outfit` để tự động gỡ bỏ miêu tả khuôn mặt nhưng giữ lại bối cảnh, trang phục.
- **Tham số khuyến nghị**:
  | Tham số | Giá trị khuyến nghị | Ý nghĩa |
  | :--- | :--- | :--- |
  | **Resolution** | `1024,1024` | Độ phân giải chuẩn cho FLUX/Krea2 |
  | **Network_Dim (Rank)** | `32` | Đủ dung lượng học chi tiết khuôn mặt & texture |
  | **Network_Alpha** | `16` | Alpha = Dim / 2 giúp hội tụ ổn định |
  | **Learning_Rate** | `1e-4` (hoặc `8e-5`) | Tốc độ học tối ưu cho AdamW8bit |
  | **Max_Train_Epochs** | `8 - 10` | Tương đương ~1500 - 2500 steps |
  | **Sample_Every_N_Steps** | `200 - 250` | Tạo ảnh mẫu định kỳ để kiểm tra quá trình học |
  | **Sample_Prompt** | `photo of [trigger_word] woman, wearing white shirt, smiling in the park, 8k uhd` | Đánh giá độ giống mặt và khả năng thay đổi trang phục |

---

### B. LoRA Xử lý Da / Retouch / Phục hồi Khuôn Mặt (Paired Control LoRA)
- **Mô hình khuyến nghị**: `FLUX.1-Kontext-dev` hoặc `Qwen-Image-Edit`
- **Engine**: Kohya Musubi-Tuner
- **Chuẩn bị dữ liệu (Ghép cặp 1-1)**:
  - `control_data`: Chứa ảnh thô / da có khuyết điểm / ảnh nén mờ (VD: `001.jpg`, `002.jpg`).
  - `train_data`: Chứa ảnh sau khi đã retouch đẹp hoàn hảo / ảnh 4K nét căng (cùng tên tệp: `001.jpg`, `002.jpg`).
- **Tham số khuyến nghị**:
  | Tham số | Giá trị khuyến nghị | Ý nghĩa |
  | :--- | :--- | :--- |
  | **Model_Type** | `FLUX.1-Kontext-dev` | Hỗ trợ kênh Control conditioning |
  | **Resolution** | `1024,1024` | Độ nét cao |
  | **Network_Dim / Alpha** | `32 / 16` | Giữ chi tiết texture da tự nhiên |
  | **Learning_Rate** | `1e-4` | |
  | **Max_Train_Epochs** | `10 - 12` | Học chuyển đổi từ Control $\rightarrow$ Target |

---

### C. LoRA Phong Cách Nghệ Thuật / Concept (Style & Art LoRA)
- **Mô hình khuyến nghị**: `FLUX.2-klein-base-9B` hoặc `Z-Image-Turbo`
- **Engine**: Kohya Musubi-Tuner
- **Chuẩn bị dữ liệu**:
  - 30 - 60 ảnh mang phong cách đồng nhất (màu nước, anime, cyberpunk, sơn dầu...).
  - Caption nên miêu tả chi tiết đối tượng trong ảnh, tránh nhắc tên phong cách trong caption (để phong cách được gắn vào trigger word).
- **Tham số khuyến nghị**:
  | Tham số | Giá trị khuyến nghị |
  | :--- | :--- |
  | **Network_Dim / Alpha** | `32 / 16` (hoặc `64 / 32` cho style phức tạp) |
  | **Learning_Rate** | `1.5e-4` |
  | **Max_Train_Epochs** | `6 - 8` |

---

### D. LoRA Video Động (Video Motion & Physics LoRA)
- **Mô hình khuyến nghị**: `Wan2.1-T2V-14B` (Text-to-Video) hoặc `Wan2.1-I2V-14B-720P` (Image-to-Video), `Wan2.2-T2V-14B`
- **Engine**: Kohya Musubi-Tuner
- **Chuẩn bị dữ liệu**:
  - 10 - 30 video clips ngắn chất lượng cao (`.mp4`), cắt đoạn 3 - 5 giây.
  - Đặt vào thư mục `TranningLorasData/datasets/train_data/my_video_dataset`.
- **Tham số khuyến nghị**:
  | Tham số | Giá trị khuyến nghị |
  | :--- | :--- |
  | **Resolution** | `720,1280` (16:9) hoặc `1280,720` (9:16) |
  | **Target_Frames** | `25` (hoặc `33`, `49` tùy VRAM) |
  | **Frame_Stride** | `1` (giữ chuyển động mượt mà) |
  | **Timestep_Sampling** | `shift` (Wan 2.1) |
  | **Timestep_Boundary** | `875` (dành riêng cho Wan 2.2) |
  | **Network_Dim / Alpha** | `32 / 16` |
  | **Max_Train_Epochs** | `15 - 20` |

---

## 📊 3. Bảng Tổng Hợp Tham Số Vàng (Cheat Sheet)

| Tác vụ Huấn luyện | Mô hình Tối ưu | Engine | Dim / Alpha | Learning Rate | Steps Khuyến nghị | VRAM GPU L4 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Nhân vật (Face/Body)** | `Krea2-Raw` | Musubi | `32 / 16` | `1e-4` | 1500 - 2000 | ~17.5 GB |
| **Nhân vật (FLUX.1)** | `FLUX.1-dev` | AI-Toolkit | `16 / 16` | `1e-4` | 1500 - 2000 | ~16.2 GB |
| **Xử lý Da / Retouch** | `FLUX.1-Kontext` | Musubi | `32 / 16` | `1e-4` | 2000 - 2500 | ~18.0 GB |
| **Phong cách Đồ họa** | `FLUX.2-klein-9B` | Musubi | `32 / 16` | `1.5e-4` | 1200 - 1800 | ~18.5 GB |
| **Tốc độ Siêu nhanh** | `Z-Image-Turbo` | Musubi | `16 / 8` | `2e-4` | 800 - 1200 | ~14.0 GB |
| **Chỉnh sửa / Inpaint** | `Qwen-Image-Edit` | Musubi | `32 / 16` | `1e-4` | 1800 - 2200 | ~18.2 GB |
| **Video Text-to-Video** | `Wan2.1-T2V-14B` | Musubi | `32 / 16` | `1e-4` | 2500 - 3500 | ~21.0 GB |
| **Video Image-to-Video**| `Wan2.1-I2V-720P`| Musubi | `32 / 16` | `1e-4` | 2500 - 3500 | ~22.5 GB |

---

## 🔐 4. Quản Lý Khóa Bảo Mật (API Key Vault) & Lưu Trữ Model Vĩnh Viễn

1. **API Key Vault Tự Động**:
   - Khi bạn nhập API Key bất kỳ (Gemini, Hugging Face, WandB, CivitAI), hệ thống tự động lưu phân loại vào `/content/drive/MyDrive/TranningLorasData/config/api_vault.json`.
   - Các lần chạy sau chỉ cần để trống ô API Key, hệ thống sẽ **tự động lấy key mặc định đã lưu**.
2. **Kho Trọng Số Vĩnh Viễn (Persistent Model Storage)**:
   - Toàn bộ Model Base (24.5 GB), VAE và Text Encoder khi tải lần đầu sẽ được lưu trực tiếp vào Google Drive.
   - Các lần chạy sau: Hệ thống quét phát hiện file đã có sẵn $\rightarrow$ **Bỏ qua bước tải 100% (0 giây chờ)**.
3. **Cơ Chế Phục Hồi Token Thông Minh (Anonymous Recovery)**:
   - Nếu bạn nhập Token Hugging Face chưa duyệt quyền, hệ thống tự động gỡ token và tải ẩn danh từ mirror công khai, ngăn ngừa hoàn toàn lỗi `401 Unauthorized`.

---

## 🛠️ 5. Danh Sách Notebooks Sẵn Sàng Chạy

- **`01_Universal_Image_LoRA_Trainer.ipynb`**: Huấn luyện LoRA hình ảnh cho FLUX.1, FLUX.2, Kontext, Krea2, Qwen, Z-Image.  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenducvuongg/TranningLoras/blob/main/notebooks/01_Universal_Image_LoRA_Trainer.ipynb)

- **`02_Universal_Video_LoRA_Trainer.ipynb`**: Huấn luyện LoRA Video Text-to-Video & Image-to-Video cho Wan 2.1 & Wan 2.2.  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenducvuongg/TranningLoras/blob/main/notebooks/02_Universal_Video_LoRA_Trainer.ipynb)

- **`03_Dataset_Captioning_Tools.ipynb`**: Công cụ gán nhãn tự động chuyên sâu với Gemini 3.6 Flash & Joy Caption.  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenducvuongg/TranningLoras/blob/main/notebooks/03_Dataset_Captioning_Tools.ipynb)

- **`04_Toolkit_WebUI_Trainer.ipynb`**: Giao diện đồ họa AI-Toolkit WebUI qua Colab Port Proxy.  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenducvuongg/TranningLoras/blob/main/notebooks/04_Toolkit_WebUI_Trainer.ipynb)
