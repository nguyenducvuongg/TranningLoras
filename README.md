# 🎨 TranningLoras: Universal Image LoRA Training Suite

> **Bộ công cụ huấn luyện LoRA Hình Ảnh Chuyên Sâu trên Google Colab & Linux GPU (Tối ưu hóa tuyệt đối cho GPU T4 16GB, L4 24GB, A100 40GB/80GB, V100)**  
> Tích hợp 3 Engine hàng đầu thế giới:
> - **Kohya `sd-scripts`**: SDXL (Pony V6, Illustrious-XL, Animagine XL), SD 1.5, SD 3.5.
> - **Kohya `musubi-tuner`**: MMDiT thế hệ mới (FLUX.2 Klein, Qwen-Image / Edit, Z-Image Turbo, Krea2).
> - **Ostris `ai-toolkit`**: FLUX.1 (dev, schnell, kontext), SDXL, DoRA/LoRA.

---

## ⚡ Mở Nhanh Trên Google Colab (One-Click Launch)

| Tên Notebook | Mô tả & Hỗ trợ Mô hình | Mở trên Colab |
| :--- | :--- | :---: |
| **01. Universal Image LoRA Trainer** | Huấn luyện LoRA Hình Ảnh đa năng: **Pony V6 XL, Illustrious-XL, Animagine XL, SDXL, SD 1.5, SD 3.5, FLUX.1, FLUX.2 Klein, Qwen-Image / Edit, Z-Image, Krea2, Custom Checkpoints** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenducvuongg/TranningLoras/blob/main/notebooks/01_Universal_Image_LoRA_Trainer.ipynb) |
| **02. Dataset Captioning Studio** | Bộ công cụ gán nhãn AI tự động chuyên sâu: **Gemini 3.7 / 3.6 / 3.5 Flash & Pro, JoyCaption (4-bit), Florence-2, OpenAI GPT-4o** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenducvuongg/TranningLoras/blob/main/notebooks/02_Dataset_Captioning_Studio.ipynb) |
| **03. Toolkit WebUI Trainer** | Giao diện đồ họa WebUI trực quan của AI-Toolkit qua Colab Port Proxy | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nguyenducvuongg/TranningLoras/blob/main/notebooks/03_Toolkit_WebUI_Trainer.ipynb) |

---

## 🛡️ Bộ Kỹ Thuật Chống "Da Nhựa / AI Gloss" & Giữ 100% Likeness

1. **`--noise_offset 0.05 - 0.07`**: Thêm dịch chuyển độ sáng trung bình vào quá trình khuếch tán, triệt tiêu hoàn toàn hiện tượng da búp bê bóng nhờn, phục hồi độ tương phản sâu và vi sắc thái tự nhiên.
2. **`--min_snr_gamma 5`**: Cân bằng lại gradient loss giữa các timestep khuếch tán, bảo toàn trọn vẹn lỗ chân lông (pores), sợi tóc tơ, nếp nhăn và viền sắc cạnh của sản phẩm.
3. **`--no_half_vae` (SDXL)**: Vận hành bộ giải mã VAE ở độ chính xác FP32 đầy đủ, tránh biến dạng màu da và hiện tượng vỡ khối màu (banding).
4. **Likeness-Preserving Captioning**: Kịch bản prompt thông minh chỉ mô tả bối cảnh, ánh sáng, trang phục và góc chụp; **tuyệt đối không miêu tả đặc trưng nhận diện khuôn mặt hay hình khối logo độc quyền**, ép 100% thông tin nhận diện được học trọn vẹn vào trigger word.

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
│   ├── dit/                         # Trọng số DiT (Klein, Kontext, Qwen, Z-Image, Krea2...)
│   ├── sdxl/                        # Trọng số SDXL (Pony, Illustrious, Animagine, Base 1.0...)
│   ├── sd15/                        # Trọng số SD 1.5 (v1-5, Realistic Vision...)
│   ├── vae/                         # Trọng số VAE (flux_vae, qwen_vae, sdxl_vae...)
│   └── text_encoders/               # CLIP-L, T5-XXL, Qwen-VL...
├── datasets/                        # Thư mục dữ liệu huấn luyện
│   ├── train_data/                  # Chứa ảnh mục tiêu (Target Data)
│   │   └── 20_my_character/         # VD: 20 ảnh nhân vật kèm file caption .txt
│   └── control_data/                # Chứa ảnh đầu vào đối chiếu cho Paired LoRA (Skin/Retouch/Upscale)
│       └── paired_skin_raw/         # VD: Ảnh da mụn/mờ có tên file khớp 1-1 với train_data
├── outputs/                         # Nơi xuất file LoRA hoàn thiện (.safetensors)
│   ├── sample_images/               # Ảnh test sinh ra qua các step
│   └── ComfyUI_Ready/               # LoRA đã tự động convert sang định dạng ComfyUI
└── engines_cache/                   # Môi trường cache wheels & packages
```

---

## 🚀 2. Hướng Dẫn Chi Tiết Huấn Luyện 6 Dạng LoRA Ảnh

### A. LoRA Khuôn Mặt Giống Thật 100% (Face Likeness & Portrait)
- **Mô hình khuyến nghị**: `SDXL-Base-1.0` hoặc `Pony-Diffusion-V6-XL`
- **Engine**: Kohya `sd-scripts`
- **Chiến lược Caption**: Chọn kịch bản `Face_Likeness`. Trigger word (ví dụ: `ohwx person`).
- **Tham số chống nhựa**: `Noise Offset = 0.06`, `Min-SNR Gamma = 5`, `No Half VAE = True`.

### B. LoRA Nhân Vật Nhất Quán (Character Consistency)
- **Mô hình khuyến nghị**: `Pony-Diffusion-V6-XL` hoặc `Illustrious-XL-v0.1`
- **Engine**: Kohya `sd-scripts`
- **Chiến lược Caption**: Chọn kịch bản `Character_Outfit` để học dáng người, trang phục và tóc qua các góc máy.

### C. LoRA Xử Lý Da Thực & Retouch (Skin Texture & Pores)
- **Mô hình khuyến nghị**: `FLUX.1-Kontext-dev` hoặc `Qwen-Image-Edit`
- **Engine**: Kohya `musubi-tuner`
- **Dữ liệu Paired Control**:
  - `control_data`: Ảnh thô, khuyết điểm, mờ (`0001.jpg`, `0002.jpg`).
  - `train_data`: Ảnh sau khi chỉnh sửa giữ trọn vi vân lỗ chân lông (`0001.jpg`, `0002.jpg`).

### D. LoRA Phong Cách Nghệ Thuật (Art Style)
- **Mô hình khuyến nghị**: `FLUX.2-klein-base-9B`, `Illustrious-XL-v0.1` hoặc `Z-Image-Turbo`
- **Chiến lược Caption**: Chọn kịch bản `Art_Style` (tuyệt đối không mô tả tên style trong text caption).

### E. LoRA Upscale & Phục Hồi Chi Tiết (Restoration)
- **Mô hình khuyến nghị**: `FLUX.1-Kontext-dev` hoặc `SDXL-Base-1.0`
- **Chiến lược**: Paired Control (Low-Res -> High-Res 4K).

### F. LoRA Sản Phẩm Thương Mại & Quảng Cáo (Product Commercials)
- **Mô hình khuyến nghị**: `SDXL-Base-1.0` hoặc `FLUX.1-dev`
- **Chiến lược Caption**: Chọn kịch bản `Product_Commercial`. Giữ nguyên hình khối, logo và bao bì.

---

## 📊 3. Bảng Tổng Hợp Tham Số Vàng (Cheat Sheet)

| Tác vụ Huấn luyện | Mô hình Tối ưu | Engine | Dim / Alpha | Noise Offset | Min-SNR | Steps Khuyến nghị | VRAM GPU T4/L4 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Face Likeness** | `SDXL-Base-1.0` / `Pony` | sdscripts | `32 / 16` | `0.06` | `5` | 1500 - 2500 | ~11.5 GB (T4 mượt) |
| **Character** | `Pony` / `Illustrious` | sdscripts | `32 / 16` | `0.05` | `5` | 1500 - 2500 | ~11.5 GB (T4 mượt) |
| **Skin Retouch** | `FLUX.1-Kontext` | Musubi | `32 / 16` | - | - | 2000 - 2500 | ~18.0 GB (L4/A100) |
| **Art Style** | `FLUX.2-klein-9B` | Musubi | `32 / 16` | - | - | 1200 - 1800 | ~18.5 GB |
| **Product** | `SDXL-Base-1.0` / `FLUX.1`| sdscripts | `32 / 16` | `0.06` | `5` | 1500 - 2000 | ~12.0 GB |
| **Fast LoRA** | `Z-Image-Turbo` | Musubi | `16 / 8` | - | - | 800 - 1200 | ~14.0 GB (T4 mượt) |

---

## 🔐 4. Quản Lý Khóa Bảo Mật (API Key Vault) & Tải Tốc Độ Cao

1. **API Key Vault Tự Động**:
   - Khi bạn nhập API Key bất kỳ (Gemini, Hugging Face, WandB, CivitAI, OpenAI), hệ thống tự động lưu vào `/content/drive/MyDrive/TranningLorasData/config/api_vault.json`.
   - Các lần chạy sau chỉ cần để trống ô API Key, hệ thống sẽ **tự động lấy key mặc định đã lưu**.
2. **Kho Trọng Số Vĩnh Viễn (Persistent Model Storage)**:
   - Toàn bộ Model Base, VAE và Text Encoder khi tải lần đầu sẽ được lưu trực tiếp vào Google Drive.
   - Các lần chạy sau: Quét phát hiện file đã có sẵn $\rightarrow$ **Bỏ qua bước tải 100% (0 giây chờ)**.
3. **Cơ Chế Mirror Công Khai (Public Fallback Mirrors)**:
   - 100% các model gated (SDXL Base, SD 3.5, FLUX.1) đều được tích hợp link mirror công khai un-gated, ngăn chặn triệt để lỗi `401 Unauthorized`.
