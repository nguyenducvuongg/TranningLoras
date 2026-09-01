"""
Dataset Renamer & Standardizer
Tự động quét và đổi tên toàn bộ hình ảnh và tệp caption (.txt) trong dataset
theo quy chuẩn thứ tự đồng bộ (ví dụ: 0001.png, 0001.txt hoặc prefix_0001.jpg, prefix_0001.txt),
đồng thời đồng bộ hóa hoàn hảo với thư mục Control (nếu có).
"""

import os
import uuid
import shutil
from typing import List, Dict, Any, Optional, Tuple, Set
from .cleaner import IMAGE_EXTENSIONS, clean_directory


def standardize_dataset_filenames(
    image_folder: str,
    control_folder: Optional[str] = None,
    prefix: str = "",
    start_index: int = 1,
    digits: int = 4,
    auto_create_txt: bool = True,
    default_caption: str = "",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Chuẩn hóa tên toàn bộ hình ảnh và caption (.txt) trong thư mục train.
    
    Quy tắc hoạt động:
    1. Quét toàn bộ ảnh hợp lệ (.png, .jpg, .jpeg, .webp, .bmp,...).
    2. Đồng bộ tệp caption (.txt) tương ứng với từng ảnh.
    3. Nếu có Control_Folder, đồng bộ đổi tên ảnh đối chiếu theo đúng cặp 1-1.
    4. Sử dụng cơ chế Đổi Tên 2 Bước (Two-Phase Renaming với UUID) để chống ghi đè hoặc xung đột file.
    5. Tự động tạo tệp .txt nếu chưa có để đảm bảo an toàn tuyệt đối cho quá trình train.
    """
    if not os.path.exists(image_folder):
        if verbose:
            print(f"⚠️ Thư mục không tồn tại: {image_folder}")
        return {"renamed_images": 0, "renamed_captions": 0, "created_captions": 0, "renamed_controls": 0}

    # Dọn dẹp rác hệ thống trước khi đổi tên
    clean_directory(image_folder)
    if control_folder and os.path.exists(control_folder):
        clean_directory(control_folder)

    # 1. Thu thập danh sách ảnh
    image_files = []
    for f in sorted(os.listdir(image_folder)):
        fpath = os.path.join(image_folder, f)
        if os.path.isfile(fpath) and not f.startswith("."):
            ext = os.path.splitext(f)[1]
            if ext in IMAGE_EXTENSIONS:
                image_files.append(f)

    if not image_files:
        if verbose:
            print(f"ℹ️ Không tìm thấy ảnh hợp lệ nào trong: {image_folder}")
        return {"renamed_images": 0, "renamed_captions": 0, "created_captions": 0, "renamed_controls": 0}

    # 2. Lập kế hoạch đổi tên (Two-Phase Renaming)
    temp_stage: List[Dict[str, Any]] = []
    
    for idx, orig_name in enumerate(image_files, start=start_index):
        base_orig, ext_orig = os.path.splitext(orig_name)
        new_base = f"{prefix}{idx:0{digits}d}" if prefix else f"{idx:0{digits}d}"
        new_img_name = f"{new_base}{ext_orig.lower()}"
        new_txt_name = f"{new_base}.txt"

        orig_img_path = os.path.join(image_folder, orig_name)
        
        # Tìm file caption tương ứng (.txt hoặc .caption)
        orig_txt_path = None
        for txt_ext in [".txt", ".caption"]:
            candidate_txt = os.path.join(image_folder, f"{base_orig}{txt_ext}")
            if os.path.exists(candidate_txt):
                orig_txt_path = candidate_txt
                break

        # Tìm control file tương ứng nếu có control folder
        ctrl_orig_paths = []
        if control_folder and os.path.exists(control_folder):
            direct_ctrl = os.path.join(control_folder, orig_name)
            if os.path.exists(direct_ctrl):
                ctrl_orig_paths.append((direct_ctrl, f"{new_base}{os.path.splitext(orig_name)[1].lower()}"))
            else:
                for c_file in sorted(os.listdir(control_folder)):
                    c_base, c_ext = os.path.splitext(c_file)
                    if c_ext in IMAGE_EXTENSIONS:
                        if c_base == base_orig:
                            ctrl_orig_paths.append((os.path.join(control_folder, c_file), f"{new_base}{c_ext.lower()}"))
                        elif c_base.startswith(f"{base_orig}_"):
                            suffix = c_base[len(base_orig):]
                            ctrl_orig_paths.append((os.path.join(control_folder, c_file), f"{new_base}{suffix}{c_ext.lower()}"))

        temp_token = uuid.uuid4().hex[:8]
        temp_img_name = f"__tmp_ren_{temp_token}_{orig_name}"
        temp_img_path = os.path.join(image_folder, temp_img_name)

        temp_txt_path = None
        if orig_txt_path:
            temp_txt_name = f"__tmp_ren_{temp_token}_{os.path.basename(orig_txt_path)}"
            temp_txt_path = os.path.join(image_folder, temp_txt_name)

        temp_ctrls = []
        for orig_c_p, target_c_name in ctrl_orig_paths:
            c_dir = os.path.dirname(orig_c_p)
            temp_c_name = f"__tmp_ren_{temp_token}_{os.path.basename(orig_c_p)}"
            temp_c_path = os.path.join(c_dir, temp_c_name)
            temp_ctrls.append((orig_c_p, temp_c_path, os.path.join(c_dir, target_c_name)))

        temp_stage.append({
            "orig_img": orig_img_path,
            "temp_img": temp_img_path,
            "final_img": os.path.join(image_folder, new_img_name),
            "orig_txt": orig_txt_path,
            "temp_txt": temp_txt_path,
            "final_txt": os.path.join(image_folder, new_txt_name),
            "controls": temp_ctrls,
        })

    # 3. Giai đoạn 1: Đổi sang tên tạm
    for item in temp_stage:
        os.rename(item["orig_img"], item["temp_img"])
        if item["orig_txt"] and item["temp_txt"]:
            os.rename(item["orig_txt"], item["temp_txt"])
        for orig_c, temp_c, _ in item["controls"]:
            os.rename(orig_c, temp_c)

    # 4. Giai đoạn 2: Đổi từ tên tạm sang tên chuẩn hóa cuối cùng
    renamed_images = 0
    renamed_captions = 0
    created_captions = 0
    renamed_controls = 0

    for item in temp_stage:
        os.rename(item["temp_img"], item["final_img"])
        renamed_images += 1

        if item["temp_txt"]:
            os.rename(item["temp_txt"], item["final_txt"])
            renamed_captions += 1
        elif auto_create_txt:
            with open(item["final_txt"], "w", encoding="utf-8") as f:
                f.write(default_caption)
            created_captions += 1

        for _, temp_c, final_c in item["controls"]:
            os.rename(temp_c, final_c)
            renamed_controls += 1

    stats = {
        "folder": image_folder,
        "renamed_images": renamed_images,
        "renamed_captions": renamed_captions,
        "created_captions": created_captions,
        "renamed_controls": renamed_controls,
    }

    if verbose:
        print(f"✨ [Chuẩn hóa Dataset] Thư mục: {os.path.basename(image_folder)}")
        print(f"   🖼️ Đã đổi tên chuẩn hóa {renamed_images} hình ảnh ({prefix}{start_index:0{digits}d} ➔ {prefix}{start_index+renamed_images-1:0{digits}d})")
        print(f"   📝 Đã đồng bộ {renamed_captions} file caption (.txt) tương ứng")
        if created_captions > 0:
            print(f"   ➕ Đã tạo mới {created_captions} file caption (.txt) mẫu")
        if renamed_controls > 0:
            print(f"   🎛️ Đã đồng bộ {renamed_controls} file ảnh đối chiếu trong thư mục Control")

    return stats


def batch_standardize_datasets(
    train_folders: str,
    control_folders: Optional[str] = None,
    prefix: str = "",
    start_index: int = 1,
    digits: int = 4,
    auto_create_txt: bool = True,
    default_caption: str = "",
) -> List[Dict[str, Any]]:
    """
    Chuẩn hóa hàng loạt danh sách thư mục train (hỗ trợ nhiều thư mục cách nhau bởi dấu phẩy).
    """
    t_dirs = [d.strip() for d in train_folders.split(",") if d.strip()]
    c_dirs = [d.strip() for d in control_folders.split(",") if d.strip()] if control_folders else []

    results = []
    for i, t_dir in enumerate(t_dirs):
        c_dir = c_dirs[i] if i < len(c_dirs) else None
        res = standardize_dataset_filenames(
            image_folder=t_dir,
            control_folder=c_dir,
            prefix=prefix,
            start_index=start_index,
            digits=digits,
            auto_create_txt=auto_create_txt,
            default_caption=default_caption,
            verbose=True,
        )
        results.append(res)

    return results
