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


if __name__ == "__main__":
    unittest.main()
