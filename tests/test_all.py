import os
import shutil
import tempfile
import unittest
from lora_trainer.config.model_registry import (
    get_model_info,
    get_preferred_engine,
    MODEL_REGISTRY,
    VAE_REGISTRY,
    TEXT_ENCODER_REGISTRY,
)
from lora_trainer.config.musubi_config import MusubiConfigBuilder, dict_to_cli_args
from lora_trainer.config.toolkit_config import ToolkitConfigBuilder
from lora_trainer.data.cleaner import clean_directory, get_supported_images, get_supported_videos
from lora_trainer.data.tag_processor import process_tags, read_text_file, process_dir_tags, add_folder_name_tags
from lora_trainer.data.dataset_builder import parse_folder_steps, build_dataset_list, check_folder_stats
from lora_trainer.utils.sampler import calculate_bucket_resolution


class TestModelRegistry(unittest.TestCase):
    def test_model_registry_keys(self):
        self.assertIn("Wan2.1-T2V-14B", MODEL_REGISTRY)
        self.assertIn("Wan2.2-I2V-14B", MODEL_REGISTRY)
        self.assertIn("FLUX.1-dev", MODEL_REGISTRY)
        self.assertIn("FLUX.2-klein-base-9B", MODEL_REGISTRY)
        self.assertIn("Qwen-Image-Edit", MODEL_REGISTRY)
        self.assertIn("Qwen-Image-Edit-2509", MODEL_REGISTRY)
        self.assertIn("Z-Image-Turbo", MODEL_REGISTRY)
        self.assertIn("Z-Image-De-Turbo", MODEL_REGISTRY)

    def test_get_model_info_fuzzy(self):
        info = get_model_info("wan22-t2v-14b")
        self.assertEqual(info["arch"], "wan22")
        self.assertTrue(info["supports_video"])
        self.assertEqual(info["default_boundary"], 875)

    def test_preferred_engine(self):
        self.assertEqual(get_preferred_engine("FLUX.1-dev"), "toolkit")
        self.assertEqual(get_preferred_engine("Wan2.1-T2V-14B"), "musubi")
        self.assertEqual(get_preferred_engine("Qwen-Image"), "musubi")
        self.assertEqual(get_preferred_engine("FLUX.2-klein-base-9B"), "musubi")


class TestConfigBuilders(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_musubi_wan22_config_builder(self):
        builder = MusubiConfigBuilder(
            model_name="Wan2.2-I2V-14B",
            output_dir=os.path.join(self.temp_dir, "output"),
            output_name="wan_lora_test",
            cache_base_dir=os.path.join(self.temp_dir, "cache"),
            weights_dir=os.path.join(self.temp_dir, "weights"),
        )
        
        toml_path = os.path.join(self.temp_dir, "dataset.toml")
        builder.build_dataset_toml(
            dataset_path=toml_path,
            resolution=[720, 1280],
            video_folders=[
                {
                    "path": self.temp_dir,
                    "repeats": 5,
                    "frame_extraction": "slide",
                    "target_frames": [25],
                    "frame_stride": 2,
                }
            ],
        )
        self.assertTrue(os.path.exists(toml_path))

        train_args = builder.build_train_args(
            dataset_config_path=toml_path,
            dit_model_path="/content/models/wan22_i2v.safetensors",
            max_train_epochs=10,
            learning_rate=2e-4,
            timestep_boundary=900,
            sample_every_n_steps=200,
            sample_prompt_file="/content/prompt.txt",
        )
        self.assertIn("--timestep_boundary 900", train_args)
        self.assertIn("--network_module networks.lora_wan", train_args)
        self.assertIn("--sample_every_n_steps 200", train_args)

        cache_latents_cmd = builder.build_cache_latents_args(
            dataset_config_path=toml_path,
            vae_path="/content/models/wan_vae.pth",
            clip_vision_path="/content/models/clip_vision.pth",
        )
        self.assertIn("wan_cache_latents.py", cache_latents_cmd)
        self.assertIn("--i2v", cache_latents_cmd)

    def test_musubi_image_dataset_captions(self):
        img_dir = os.path.join(self.temp_dir, "train_imgs")
        os.makedirs(img_dir, exist_ok=True)
        img_file = os.path.join(img_dir, "sample_1.png")
        with open(img_file, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n")

        builder = MusubiConfigBuilder(
            model_name="Krea2-Raw",
            output_dir=os.path.join(self.temp_dir, "output"),
            output_name="krea_test",
        )
        toml_path = os.path.join(self.temp_dir, "dataset.toml")
        builder.build_dataset_toml(
            dataset_path=toml_path,
            resolution=[1024, 1024],
            image_folders=[{"path": img_dir, "repeats": 1}],
            caption_extension=".txt",
        )
        self.assertTrue(os.path.exists(toml_path))
        with open(toml_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn('caption_extension = ".txt"', content)

        # Check that missing txt file was automatically created
        txt_file = os.path.join(img_dir, "sample_1.txt")
        self.assertTrue(os.path.exists(txt_file))

    def test_musubi_flux2_and_qwen_config(self):
        builder_qwen = MusubiConfigBuilder(
            model_name="Qwen-Image-Edit-2509",
            output_dir=os.path.join(self.temp_dir, "output_qwen"),
            output_name="qwen_lora",
        )
        cache_latents_qwen = builder_qwen.build_cache_latents_args(
            dataset_config_path="/content/dataset.toml",
            vae_path="/content/models/qwen_vae.safetensors",
        )
        self.assertIn("qwen_image_cache_latents.py", cache_latents_qwen)
        self.assertIn("--model_version qwen_image_edit_2509", cache_latents_qwen)

    def test_toolkit_config_builder(self):
        builder = ToolkitConfigBuilder(
            model_name="FLUX.1-dev",
            output_dir=os.path.join(self.temp_dir, "output"),
            output_name="flux_lora_test",
        )
        yaml_path = os.path.join(self.temp_dir, "config.yaml")
        builder.build_yaml_config(
            save_yaml_path=yaml_path,
            dataset_folders=[
                {
                    "path": self.temp_dir,
                    "resolution": [1024],
                    "control_path": os.path.join(self.temp_dir, "ctrl"),
                }
            ],
            steps=1500,
            save_every=300,
            trigger_word="cyberpunk_girl",
            quantize=True,
        )
        self.assertTrue(os.path.exists(yaml_path))
        with open(yaml_path, "r") as f:
            content = f.read()
            self.assertIn("cyberpunk_girl", content)
            self.assertIn("qfloat8", content)


    def test_calculate_bucket_resolution(self):
        res = calculate_bucket_resolution("/non_existent.jpg", max_size=1536, divisible_by=64)
        self.assertEqual(res, (1024, 1024))


class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cleaner_and_tags(self):
        junk_file = os.path.join(self.temp_dir, ".DS_Store")
        with open(junk_file, "w") as f:
            f.write("junk")
        
        txt_file = os.path.join(self.temp_dir, "test.txt")
        with open(txt_file, "w") as f:
            f.write("a beautiful landscape")

        removed, valid = clean_directory(self.temp_dir)
        self.assertEqual(removed, 1)
        self.assertFalse(os.path.exists(junk_file))

        # Test Tag Prepend
        process_tags(txt_file, custom_tag="anime style", append=False)
        content = read_text_file(txt_file)
        self.assertEqual(content, "anime style, a beautiful landscape")

        # Test Tag Append
        process_tags(txt_file, custom_tag="masterpiece", append=True)
        content = read_text_file(txt_file)
        self.assertEqual(content, "anime style, a beautiful landscape, masterpiece")

        # Test Tag Remove
        process_tags(txt_file, custom_tag="anime style", remove_tag=True)
        content = read_text_file(txt_file)
        self.assertEqual(content, "a beautiful landscape, masterpiece")

    def test_parse_folder_steps(self):
        steps, repeats, name = parse_folder_steps("/path/to/500_cyberpunk_girl")
        self.assertEqual(steps, 500)
        self.assertEqual(repeats, 500)
        self.assertEqual(name, "cyberpunk_girl")

    def test_build_dataset_list(self):
        # Create subfolders with images
        sub1 = os.path.join(self.temp_dir, "100_concept_a")
        sub2 = os.path.join(self.temp_dir, "200_concept_b")
        os.makedirs(sub1, exist_ok=True)
        os.makedirs(sub2, exist_ok=True)

        with open(os.path.join(sub1, "img1.png"), "w") as f:
            f.write("fake")
        with open(os.path.join(sub2, "img2.jpg"), "w") as f:
            f.write("fake")

        ds_list = build_dataset_list(f"{sub1}, {sub2}")
        self.assertEqual(len(ds_list), 2)
        self.assertEqual(ds_list[0]["name"], "concept_a")
        self.assertEqual(ds_list[0]["steps"], 100)
        self.assertEqual(ds_list[1]["name"], "concept_b")
class TestKeyManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.vault_file = os.path.join(self.temp_dir, "test_vault.json")
        import lora_trainer.caption.key_manager as km
        self.original_primary = km.PRIMARY_VAULT_PATH
        self.original_fallback = km.FALLBACK_VAULT_PATHS
        km.PRIMARY_VAULT_PATH = self.vault_file
        km.FALLBACK_VAULT_PATHS = [self.vault_file]

    def tearDown(self):
        import lora_trainer.caption.key_manager as km
        km.PRIMARY_VAULT_PATH = self.original_primary
        km.FALLBACK_VAULT_PATHS = self.original_fallback
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_get_api_key(self):
        from lora_trainer.caption.key_manager import save_api_key, get_api_key, get_saved_keys_list, mask_key

        self.assertEqual(mask_key("AIzaSy1234567890abcdef"), "AIzaSy...cdef")

        # Save Gemini Key
        save_api_key("gemini", "AIzaSy_test_key_1", label="Key Main", set_default=True)
        self.assertEqual(get_api_key("gemini"), "AIzaSy_test_key_1")

        # Save second Gemini Key
        save_api_key("gemini", "AIzaSy_test_key_2", label="Key Backup", set_default=False)
        self.assertEqual(get_api_key("gemini"), "AIzaSy_test_key_1")

        # Select second key by label
        self.assertEqual(get_api_key("gemini", user_provided_key="Key Backup"), "AIzaSy_test_key_2")

        # Verify list of saved keys
        keys_list = get_saved_keys_list("gemini")
        self.assertEqual(len(keys_list), 2)
        self.assertTrue(keys_list[0]["is_default"])
        self.assertFalse(keys_list[1]["is_default"])

        # Auto-save when passing new key
        new_k = get_api_key("huggingface", user_provided_key="hf_test_token_12345", auto_save=True)
        self.assertEqual(new_k, "hf_test_token_12345")
        self.assertEqual(get_api_key("huggingface"), "hf_test_token_12345")


class TestModelStorage(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.drive_root = os.path.join(self.temp_dir, "TranningLorasData")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_setup_storage_structure_safe(self):
        from lora_trainer.engine.model_storage import setup_storage_structure

        folders = setup_storage_structure(self.drive_root)
        self.assertTrue(os.path.exists(folders["train_data"]))
        self.assertTrue(os.path.exists(folders["control_data"]))
        self.assertTrue(os.path.exists(folders["models_dit"]))
        self.assertTrue(os.path.exists(folders["models_vae"]))
        self.assertTrue(os.path.exists(folders["outputs_comfy"]))

        # Create a test file in train_data
        test_file = os.path.join(folders["train_data"], "test_image.txt")
        with open(test_file, "w") as f:
            f.write("test_content_keep_safe")

        # Run setup_storage_structure again (second time)
        folders_again = setup_storage_structure(self.drive_root)
        # Verify file is not deleted or overwritten
        self.assertTrue(os.path.exists(test_file))
        with open(test_file, "r") as f:
            self.assertEqual(f.read(), "test_content_keep_safe")

    def test_file_completeness_and_scan(self):
        from lora_trainer.engine.model_storage import is_file_complete, scan_model_suite

        test_f = os.path.join(self.temp_dir, "test_weight.safetensors")
        with open(test_f, "wb") as f:
            f.write(b"0" * (1024 * 1024 * 2))  # 2MB

        self.assertTrue(is_file_complete(test_f))

        # Check with aria2 temporary file present
        aria_f = f"{test_f}.aria2"
        with open(aria_f, "w") as f:
            f.write("temp")
        self.assertFalse(is_file_complete(test_f))
        os.remove(aria_f)

        scan = scan_model_suite("Krea2-Raw", base_dir=self.drive_root, local_dir=self.temp_dir)
        self.assertEqual(scan["model_name"], "Krea2-Raw")
        self.assertGreater(len(scan["components"]), 0)


class TestEnvironmentSetup(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_accelerate_config_generation(self):
        from lora_trainer.engine.environment_setup import setup_accelerate_config, apply_performance_environment_vars
        import yaml

        custom_cfg = os.path.join(self.temp_dir, "accel_cfg.yaml")
        setup_accelerate_config(custom_path=custom_cfg, mixed_precision="bf16")
        self.assertTrue(os.path.exists(custom_cfg))

        with open(custom_cfg, "r") as f:
            data = yaml.safe_load(f)
            self.assertEqual(data["mixed_precision"], "bf16")
            self.assertEqual(data["compute_environment"], "LOCAL_MACHINE")

        apply_performance_environment_vars(musubi_dir=self.temp_dir, toolkit_dir=self.temp_dir)
        self.assertEqual(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "expandable_segments:True")


class TestDownloader(unittest.TestCase):
    def test_aria2_progress_parser(self):
        from lora_trainer.engine.downloader import parse_aria2_progress, parse_aria2_size, prepare_download_url

        self.assertEqual(parse_aria2_size("16.2GiB"), int(16.2 * 1024**3))
        self.assertEqual(parse_aria2_size("320MiB"), int(320 * 1024**2))
        self.assertEqual(parse_aria2_size("500KiB"), int(500 * 1024))

        sample_line = "[#20982e 3.00GiB/9.12GiB(33%) CN:4 DL:183MiB ETA:35s]"
        res = parse_aria2_progress(sample_line)
        self.assertIsNotNone(res)
        self.assertEqual(res["percent"], 33)
        self.assertEqual(res["downloaded_bytes"], int(3.0 * 1024**3))
        self.assertEqual(res["total_bytes"], int(9.12 * 1024**3))
        self.assertEqual(res["speed"], "183MiB")

        hf_url = "https://huggingface.co/User/Repo/blob/main/model.safetensors"
        converted = prepare_download_url(hf_url)
        self.assertIn("/resolve/main/", converted)


class TestRenamer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.train_dir = os.path.join(self.temp_dir, "train_data")
        self.ctrl_dir = os.path.join(self.temp_dir, "control_data")
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.ctrl_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_standardize_dataset_filenames(self):
        from lora_trainer.data.renamer import standardize_dataset_filenames, batch_standardize_datasets

        # Tạo file ảnh và caption lộn xộn
        f1_img = os.path.join(self.train_dir, "photo_b (2).png")
        f1_txt = os.path.join(self.train_dir, "photo_b (2).txt")
        f2_img = os.path.join(self.train_dir, "photo_a (1).jpg")

        with open(f1_img, "wb") as f: f.write(b"fake_png_data")
        with open(f1_txt, "w", encoding="utf-8") as f: f.write("a cute cat sitting on a table")
        with open(f2_img, "wb") as f: f.write(b"fake_jpg_data")

        # Tạo paired control image
        c1_img = os.path.join(self.ctrl_dir, "photo_b (2).png")
        with open(c1_img, "wb") as f: f.write(b"fake_ctrl_data")

        stats = standardize_dataset_filenames(
            image_folder=self.train_dir,
            control_folder=self.ctrl_dir,
            prefix="img_",
            digits=4,
            auto_create_txt=True,
            default_caption="default caption test",
        )

        self.assertEqual(stats["renamed_images"], 2)
        self.assertEqual(stats["renamed_captions"], 1)
        self.assertEqual(stats["created_captions"], 1)
        self.assertEqual(stats["renamed_controls"], 1)

        # Kiểm tra file đã được đổi tên chuẩn hóa
        self.assertTrue(os.path.exists(os.path.join(self.train_dir, "img_0001.jpg")))
        self.assertTrue(os.path.exists(os.path.join(self.train_dir, "img_0001.txt")))
        self.assertTrue(os.path.exists(os.path.join(self.train_dir, "img_0002.png")))
        self.assertTrue(os.path.exists(os.path.join(self.train_dir, "img_0002.txt")))

        # Kiểm tra nội dung caption được giữ nguyên
        with open(os.path.join(self.train_dir, "img_0002.txt"), "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "a cute cat sitting on a table")

        # Kiểm tra caption mẫu được tạo mới
        with open(os.path.join(self.train_dir, "img_0001.txt"), "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "default caption test")

        # Kiểm tra control file được đổi tên đồng bộ
        self.assertTrue(os.path.exists(os.path.join(self.ctrl_dir, "img_0002.png")))


if __name__ == "__main__":
    unittest.main()
