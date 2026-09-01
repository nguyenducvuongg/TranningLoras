import os
import shutil
import tempfile
import unittest

from lora_trainer.core.model_registry import (
    get_model_info,
    get_preferred_engine,
    list_supported_models,
    MODEL_REGISTRY,
    VAE_REGISTRY,
    TEXT_ENCODER_REGISTRY,
)
from lora_trainer.core.hardware import detect_hardware_environment
from lora_trainer.core.key_vault import save_api_key, get_api_key, load_api_vault, mask_key
from lora_trainer.storage.drive_manager import setup_storage_structure, is_file_complete, get_model_component_paths
from lora_trainer.storage.downloader import prepare_download_url
from lora_trainer.dataset.cleaner import clean_directory, get_supported_images, get_supported_videos
from lora_trainer.dataset.renamer import standardize_single_folder, batch_standardize_datasets
from lora_trainer.dataset.tagger import process_tags, read_text_file, process_dir_tags, add_folder_name_tags
from lora_trainer.dataset.builder import parse_folder_steps, build_dataset_list, check_folder_stats
from lora_trainer.dataset.video_tools import calculate_bucket_resolution
from lora_trainer.captioning.base_captioner import build_task_prompt
from lora_trainer.configs.sdscripts_config import SdScriptsConfigBuilder
from lora_trainer.configs.musubi_config import MusubiConfigBuilder, dict_to_cli_args
from lora_trainer.configs.toolkit_config import ToolkitConfigBuilder


class TestModelRegistry(unittest.TestCase):
    def test_model_registry_keys(self):
        # SDXL
        self.assertIn("SDXL-Base-1.0", MODEL_REGISTRY)
        self.assertIn("Pony-Diffusion-V6-XL", MODEL_REGISTRY)
        self.assertIn("Illustrious-XL-v0.1", MODEL_REGISTRY)
        self.assertIn("Animagine-XL-3.1", MODEL_REGISTRY)
        # SD 1.5
        self.assertIn("v1-5-pruned-emaonly", MODEL_REGISTRY)
        self.assertIn("Realistic-Vision-v5.1", MODEL_REGISTRY)
        # SD 3.5
        self.assertIn("SD3.5-Large", MODEL_REGISTRY)
        # Flux & Video
        self.assertIn("Wan2.1-T2V-14B", MODEL_REGISTRY)
        self.assertIn("Wan2.2-I2V-14B", MODEL_REGISTRY)
        self.assertIn("FLUX.1-dev", MODEL_REGISTRY)
        self.assertIn("FLUX.2-klein-base-9B", MODEL_REGISTRY)
        self.assertIn("Qwen-Image-Edit", MODEL_REGISTRY)
        self.assertIn("Z-Image-Turbo", MODEL_REGISTRY)
        self.assertIn("Krea2-Raw", MODEL_REGISTRY)

    def test_get_model_info_fuzzy(self):
        info = get_model_info("pony-diffusion-v6-xl")
        self.assertEqual(info["arch"], "sdxl")
        self.assertEqual(info["engine"], "sdscripts")

        info_wan = get_model_info("wan22-t2v-14b")
        self.assertEqual(info_wan["arch"], "wan22")
        self.assertTrue(info_wan["supports_video"])

    def test_preferred_engine(self):
        self.assertEqual(get_preferred_engine("Pony-Diffusion-V6-XL"), "sdscripts")
        self.assertEqual(get_preferred_engine("SDXL-Base-1.0"), "sdscripts")
        self.assertEqual(get_preferred_engine("FLUX.1-dev"), "toolkit")
        self.assertEqual(get_preferred_engine("Wan2.1-T2V-14B"), "musubi")
        self.assertEqual(get_preferred_engine("Qwen-Image"), "musubi")

    def test_custom_model_resolution(self):
        custom_info = get_model_info("Custom-Model", custom_download_url="https://civitai.com/api/download/models/12345")
        self.assertEqual(custom_info["download_url"], "https://civitai.com/api/download/models/12345")
        self.assertEqual(custom_info["arch"], "sdxl")


class TestHardwareAndVault(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.vault_file = os.path.join(self.temp_dir, "test_vault.json")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_hardware_detection(self):
        hw = detect_hardware_environment()
        self.assertIn("gpu_name", hw)
        self.assertIn("recommended_precision", hw)
        self.assertIn("recommended_batch_size", hw)

    def test_key_vault_save_load(self):
        save_api_key("gemini", "AIzaSyTestKey123", label="test_key", vault_path=self.vault_file)
        retrieved = get_api_key("gemini", vault_path=self.vault_file)
        self.assertEqual(retrieved, "AIzaSyTestKey123")
        self.assertEqual(mask_key(retrieved), "AIza...y123")


class TestStorageAndDownloader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_setup_storage_structure(self):
        folders = setup_storage_structure(self.temp_dir)
        self.assertTrue(os.path.exists(folders["models_dit"]))
        self.assertTrue(os.path.exists(folders["train_data"]))
        self.assertTrue(os.path.exists(folders["outputs"]))

    def test_prepare_download_url(self):
        hf_blob = "https://huggingface.co/user/repo/blob/main/model.safetensors"
        hf_resolved = prepare_download_url(hf_blob)
        self.assertIn("/resolve/main/", hf_resolved)

        civitai_url = "https://civitai.com/api/download/models/12345"
        with_token = prepare_download_url(civitai_url, civitai_key="my_civit_token")
        self.assertIn("token=my_civit_token", with_token)


class TestConfigBuilders(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sdscripts_sdxl_config(self):
        builder = SdScriptsConfigBuilder(
            model_name="Pony-Diffusion-V6-XL",
            model_path="/content/models/pony.safetensors",
            output_dir=os.path.join(self.temp_dir, "output"),
            output_name="pony_lora",
            arch="sdxl",
        )
        toml_path = os.path.join(self.temp_dir, "sd_dataset.toml")
        builder.build_dataset_toml(
            dataset_path=toml_path,
            image_folders=[{"path": self.temp_dir, "repeats": 10}],
            resolution=1024,
        )
        self.assertTrue(os.path.exists(toml_path))

        train_cmd = builder.build_train_args(
            dataset_config_path=toml_path,
            learning_rate=1e-4,
            network_dim=32,
            network_alpha=16,
            max_train_epochs=10,
        )
        self.assertIn("sdxl_train_network.py", train_cmd)
        self.assertIn("--network_dim 32", train_cmd)
        self.assertIn("--network_alpha 16", train_cmd)

    def test_musubi_wan_config(self):
        builder = MusubiConfigBuilder(
            model_name="Wan2.1-T2V-14B",
            output_dir=os.path.join(self.temp_dir, "output"),
            output_name="wan_lora",
        )
        toml_path = os.path.join(self.temp_dir, "musubi_dataset.toml")
        builder.build_dataset_toml(
            dataset_path=toml_path,
            resolution=[720, 1280],
            video_folders=[{"path": self.temp_dir, "repeats": 5}],
        )
        self.assertTrue(os.path.exists(toml_path))

        train_cmd = builder.build_train_args(
            dataset_config_path=toml_path,
            dit_model_path="/content/models/wan.safetensors",
            learning_rate=1e-4,
        )
        self.assertIn("wan_train_network.py", train_cmd)
        self.assertIn("--network_module networks.lora_wan", train_cmd)

    def test_toolkit_yaml_config(self):
        builder = ToolkitConfigBuilder(
            model_name="FLUX.1-dev",
            output_dir=os.path.join(self.temp_dir, "output"),
            output_name="flux_lora",
        )
        yaml_path = os.path.join(self.temp_dir, "toolkit_config.yaml")
        builder.build_yaml_config(
            save_yaml_path=yaml_path,
            dataset_folders=[{"path": self.temp_dir}],
            steps=1500,
        )
        self.assertTrue(os.path.exists(yaml_path))


class TestDatasetAndCaptioning(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tag_processor(self):
        original = "solo, girl, dress, smile"
        processed = process_tags(original, trigger_word="my_trigger", position="prefix")
        self.assertTrue(processed.startswith("my_trigger, solo, girl"))

    def test_folder_steps_parsing(self):
        repeats, concept = parse_folder_steps("/path/to/25_anime_girl")
        self.assertEqual(repeats, 25)
        self.assertEqual(concept, "anime_girl")

    def test_bucket_resolution_calc(self):
        w, h = calculate_bucket_resolution(1920, 1080, max_pixels=1024*1024)
        self.assertEqual(w % 64, 0)
        self.assertEqual(h % 64, 0)

    def test_build_task_prompt(self):
        prompt = build_task_prompt("Skin_Portrait", "Short", trigger_word="civit_face")
        self.assertIn("civit_face", prompt)
        self.assertIn("skin", prompt.lower())


class TestBackwardCompatibility(unittest.TestCase):
    def test_legacy_imports(self):
        # 1. engine.downloader
        from lora_trainer.engine.downloader import download_model_suite, aria2_download
        self.assertTrue(callable(download_model_suite))
        self.assertTrue(callable(aria2_download))

        # 2. engine runners
        from lora_trainer.engine.musubi_runner import run_musubi_pipeline
        from lora_trainer.engine.toolkit_runner import run_toolkit_pipeline
        self.assertTrue(callable(run_musubi_pipeline))
        self.assertTrue(callable(run_toolkit_pipeline))

        # 3. config shims
        from lora_trainer.config.model_registry import get_model_info
        from lora_trainer.config.musubi_config import MusubiConfigBuilder
        from lora_trainer.config.toolkit_config import ToolkitConfigBuilder
        self.assertTrue(callable(get_model_info))

        # 4. data shims
        from lora_trainer.data.cleaner import clean_directory
        from lora_trainer.data.dataset_builder import build_dataset_list
        from lora_trainer.data.renamer import batch_standardize_datasets
        from lora_trainer.data.tag_processor import process_tags

        # 5. caption shims
        from lora_trainer.caption.key_manager import save_api_key, get_api_key
        from lora_trainer.caption.gemini_captioner import batch_caption_gemini

        # 6. utils shims
        from lora_trainer.utils.converter import auto_convert_checkpoints
        from lora_trainer.utils.sampler import get_random_sample_prompt
        from lora_trainer.utils.colab_utils import auto_disconnect


if __name__ == "__main__":
    unittest.main()
