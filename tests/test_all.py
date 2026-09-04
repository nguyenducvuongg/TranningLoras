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
from lora_trainer.dataset.cleaner import clean_directory, get_supported_images
from lora_trainer.dataset.renamer import standardize_single_folder, batch_standardize_datasets
from lora_trainer.dataset.tagger import process_tags, read_text_file, process_dir_tags, add_folder_name_tags
from lora_trainer.dataset.builder import parse_folder_steps, build_dataset_list, check_folder_stats, calculate_bucket_resolution
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
        # Flux & Next-Gen DiTs
        self.assertIn("FLUX.1-dev", MODEL_REGISTRY)
        self.assertIn("FLUX.2-klein-base-9B", MODEL_REGISTRY)
        self.assertIn("Qwen-Image", MODEL_REGISTRY)
        self.assertIn("Qwen-Image-Edit", MODEL_REGISTRY)
        self.assertIn("Z-Image-Turbo", MODEL_REGISTRY)
        self.assertIn("Krea2-Raw", MODEL_REGISTRY)
        # Verify Video Models are removed
        self.assertNotIn("Wan2.1-T2V-14B", MODEL_REGISTRY)
        self.assertNotIn("Wan2.2-I2V-14B", MODEL_REGISTRY)

    def test_get_model_info_fuzzy(self):
        info = get_model_info("pony-diffusion-v6-xl")
        self.assertEqual(info["arch"], "sdxl")
        self.assertEqual(info["engine"], "sdscripts")

        info_qwen = get_model_info("qwen-image-edit")
        self.assertEqual(info_qwen["arch"], "qwen_image_edit")
        self.assertEqual(info_qwen["type"], "image")

    def test_preferred_engine(self):
        self.assertEqual(get_preferred_engine("Pony-Diffusion-V6-XL"), "sdscripts")
        self.assertEqual(get_preferred_engine("SDXL-Base-1.0"), "sdscripts")
        self.assertEqual(get_preferred_engine("FLUX.1-dev"), "toolkit")
        self.assertEqual(get_preferred_engine("Qwen-Image"), "musubi")
        self.assertEqual(get_preferred_engine("FLUX.2-klein-base-9B"), "musubi")

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
        self.assertIn("recommended_noise_offset", hw)
        self.assertIn("recommended_min_snr_gamma", hw)

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
            noise_offset=0.06,
            min_snr_gamma=5,
            no_half_vae=True,
        )
        self.assertIn("sdxl_train_network.py", train_cmd)
        self.assertIn("--network_dim 32", train_cmd)
        self.assertIn("--network_alpha 16", train_cmd)
        self.assertIn("--sdpa", train_cmd)
        self.assertIn("--noise_offset 0.06", train_cmd)
        self.assertIn("--min_snr_gamma 5", train_cmd)
        self.assertIn("--no_half_vae", train_cmd)

    def test_musubi_dit_config(self):
        builder = MusubiConfigBuilder(
            model_name="FLUX.2-klein-base-9B",
            output_dir=os.path.join(self.temp_dir, "output"),
            output_name="flux2_lora",
        )
        toml_path = os.path.join(self.temp_dir, "musubi_dataset.toml")
        builder.build_dataset_toml(
            dataset_path=toml_path,
            resolution=[1024, 1024],
            image_folders=[{"path": self.temp_dir, "repeats": 5}],
        )
        self.assertTrue(os.path.exists(toml_path))
        with open(toml_path, "r") as f:
            toml_content = f.read()
        self.assertIn("image_directory", toml_content)
        self.assertNotIn("image_dir =", toml_content)

        train_cmd = builder.build_train_args(
            dataset_config_path=toml_path,
            dit_model_path="/content/models/flux2.safetensors",
            learning_rate=1e-4,
        )
        self.assertIn("--network_module networks.lora_flux", train_cmd)
        self.assertIn("--sdpa", train_cmd)

        # Test self-healing TOML sanitizer
        from lora_trainer.engines.musubi_engine import sanitize_musubi_toml_from_command, sanitize_musubi_train_command
        bad_toml = os.path.join(self.temp_dir, "bad.toml")
        with open(bad_toml, "w") as f:
            f.write("[[datasets]]\nimage_dir = '/test/dir'\ncontrol_image_dir = '/test/ctrl'\n")
        sanitize_musubi_toml_from_command(f"python cache.py --dataset_config '{bad_toml}'")
        with open(bad_toml, "r") as f:
            fixed_content = f.read()
        self.assertIn("image_directory = '/test/dir'", fixed_content)
        self.assertIn("control_directory = '/test/ctrl'", fixed_content)

        # Test Krea 2 fp8_scaled auto-inclusion
        krea_builder = MusubiConfigBuilder(
            model_name="Krea2-Raw",
            output_dir=os.path.join(self.temp_dir, "output"),
            output_name="krea_lora",
        )
        krea_cmd = krea_builder.build_train_args(
            dataset_config_path="/content/dataset.toml",
            dit_model_path="/content/krea2.safetensors",
            fp8_base=True,
        )
        self.assertIn("--fp8_base", krea_cmd)
        self.assertIn("--fp8_scaled", krea_cmd)

        # Test train_cmd sanitizer auto-injection
        raw_cmd = "accelerate launch src/musubi_tuner/krea2_train_network.py --fp8_base --dit test.safetensors"
        sanitized_cmd = sanitize_musubi_train_command(raw_cmd)
        self.assertIn("--fp8_base --fp8_scaled", sanitized_cmd)

        # Test safe sample_prompts with vae
        dummy_vae = os.path.join(self.temp_dir, "vae.safetensors")
        dummy_te = os.path.join(self.temp_dir, "te.safetensors")
        dummy_sample = os.path.join(self.temp_dir, "sample.txt")
        open(dummy_vae, "w").close()
        open(dummy_te, "w").close()
        open(dummy_sample, "w").close()

        cmd_with_vae = krea_builder.build_train_args(
            dataset_config_path="/content/dataset.toml",
            dit_model_path="/content/krea2.safetensors",
            vae_path=dummy_vae,
            text_encoder_path=dummy_te,
            sample_prompt_file=dummy_sample,
            sample_every_n_steps=100,
        )
        self.assertIn(f"--vae '{dummy_vae}'", cmd_with_vae)
        self.assertIn(f"--sample_prompts '{dummy_sample}'", cmd_with_vae)

        # Test sanitizer strips sample_prompts if vae is missing
        unsafe_sample_cmd = "accelerate launch krea2_train_network.py --sample_prompts 'sample.txt' --sample_every_n_steps 100"
        safe_stripped_cmd = sanitize_musubi_train_command(unsafe_sample_cmd)
        self.assertNotIn("--sample_prompts", safe_stripped_cmd)
        self.assertNotIn("--sample_every_n_steps", safe_stripped_cmd)

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
        prompt_skin = build_task_prompt("Skin_Portrait", "Short", trigger_word="civit_face")
        self.assertIn("civit_face", prompt_skin)
        self.assertIn("skin", prompt_skin.lower())

        prompt_face = build_task_prompt("Face_Likeness", "Medium", trigger_word="my_char")
        self.assertIn("my_char", prompt_face)
        self.assertIn("likeness", prompt_face.lower())

        prompt_prod = build_task_prompt("Product_Commercial", "Long", trigger_word="brand_item")
        self.assertIn("brand_item", prompt_prod)
        self.assertIn("product", prompt_prod.lower())

    def test_generate_accelerate_config(self):
        from lora_trainer.core.hardware import generate_accelerate_config
        cfg_path = os.path.join(self.temp_dir, "accelerate.yaml")
        generate_accelerate_config(output_path=cfg_path, precision="bf16", num_processes=1)
        self.assertTrue(os.path.exists(cfg_path))
        with open(cfg_path, "r") as f:
            content = f.read()
        self.assertIn("mixed_precision: bf16", content)
        self.assertIn("num_processes: 1", content)


class TestBackwardCompatibility(unittest.TestCase):
    def test_legacy_imports(self):
        # 1. engine.downloader
        from lora_trainer.engine.downloader import download_model_suite, aria2_download
        self.assertTrue(callable(download_model_suite))
        self.assertTrue(callable(aria2_download))

        # 2. engine runners
        import inspect
        from lora_trainer.engine.musubi_runner import run_musubi_pipeline
        from lora_trainer.engine.toolkit_runner import run_toolkit_pipeline
        from lora_trainer.engines.sdscripts_engine import run_sdscripts_pipeline
        self.assertTrue(callable(run_musubi_pipeline))
        self.assertTrue(callable(run_toolkit_pipeline))
        self.assertTrue(callable(run_sdscripts_pipeline))
        self.assertIn("dashboard", inspect.signature(run_musubi_pipeline).parameters)
        self.assertIn("dashboard", inspect.signature(run_sdscripts_pipeline).parameters)
        self.assertIn("dashboard", inspect.signature(run_toolkit_pipeline).parameters)

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


class TestTrainingDashboard(unittest.TestCase):
    def test_dashboard_parsing_and_stages(self):
        from lora_trainer.ui.dashboard import TrainingDashboard, create_dashboard, get_dashboard

        dash = create_dashboard(
            model_name="Krea2-Raw",
            engine_name="musubi",
            task_type="face",
            lora_name="test_lora",
            total_steps=1000,
            total_epochs=10,
        )
        self.assertEqual(get_dashboard(), dash)
        self.assertEqual(dash.status, "INITIALIZING")

        # Test stage changes
        dash.set_stage(1, "running", "Pre-cache VAE")
        self.assertEqual(dash.stages[0]["status"], "running")

        dash.set_stage(2, "running", "Pre-cache Text")
        self.assertEqual(dash.stages[0]["status"], "done")
        self.assertEqual(dash.stages[1]["status"], "running")

        # Test line parsing
        tqdm_line = "steps:  25%|██▌       | 250/1000 [03:15<09:45,  1.28it/s, loss=0.0782]"
        dash.parse_line(tqdm_line)
        self.assertEqual(dash.percent, 25.0)
        self.assertEqual(dash.current_step, 250)
        self.assertEqual(dash.total_steps, 1000)
        self.assertEqual(dash.speed, "1.28it/s")
        self.assertAlmostEqual(dash.current_loss, 0.0782, places=4)

        # Test sparkline generation
        dash.parse_line("loss=0.0650")
        dash.parse_line("loss=0.0520")
        sparkline = dash._generate_sparkline()
        self.assertIn("<svg", sparkline)
        self.assertIn("<polyline", sparkline)

        # Test completion
        dash.finish(success=True)
        self.assertEqual(dash.status, "COMPLETED")
        self.assertEqual(dash.percent, 100.0)


class TestAllRegisteredModels(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_all_29_registered_models_generate_valid_configs(self):
        from lora_trainer.core.model_registry import MODEL_REGISTRY
        from lora_trainer.configs.musubi_config import MusubiConfigBuilder
        from lora_trainer.configs.sdscripts_config import SdScriptsConfigBuilder
        from lora_trainer.configs.toolkit_config import ToolkitConfigBuilder

        for name, info in MODEL_REGISTRY.items():
            engine = info.get("engine")
            arch = info.get("arch", "sdxl")

            if engine == "musubi":
                b = MusubiConfigBuilder(name, self.temp_dir, "test_lora")
                toml_path = os.path.join(self.temp_dir, f"{name}.toml")
                b.build_dataset_toml(toml_path, image_folders=[{"path": self.temp_dir}])
                cmd = b.build_train_args(toml_path, os.path.join(self.temp_dir, "model.safetensors"))
                self.assertTrue(len(cmd) > 0)
                if arch == "krea2":
                    self.assertIn("--fp8_scaled", cmd)
                elif "qwen" in arch:
                    self.assertIn("networks.lora_qwen_image", cmd)
                elif arch == "z_image":
                    self.assertIn("networks.lora_zimage", cmd)

            elif engine == "sdscripts":
                b = SdScriptsConfigBuilder(name, os.path.join(self.temp_dir, "model.safetensors"), self.temp_dir, "test_lora", arch=arch)
                toml_path = os.path.join(self.temp_dir, f"{name}.toml")
                b.build_dataset_toml(toml_path, image_folders=[{"path": self.temp_dir}])
                cmd = b.build_train_args(toml_path)
                self.assertTrue(len(cmd) > 0)
                if arch == "sdxl":
                    self.assertIn("sdxl_train_network.py", cmd)
                elif arch in ["sd35", "sd3"]:
                    self.assertIn("sd3_train_network.py", cmd)
                elif arch == "sd15":
                    self.assertIn("train_network.py", cmd)

            elif engine == "toolkit":
                b = ToolkitConfigBuilder(name, self.temp_dir, "test_lora", model_path=os.path.join(self.temp_dir, "model.safetensors"))
                yaml_path = os.path.join(self.temp_dir, f"{name}.yaml")
                b.build_yaml_config(yaml_path, dataset_folders=[{"path": self.temp_dir}])
                self.assertTrue(os.path.exists(yaml_path))


if __name__ == "__main__":
    unittest.main()
