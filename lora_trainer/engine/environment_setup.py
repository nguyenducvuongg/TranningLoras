"""
Backward-compatible shim for lora_trainer.engine.environment_setup.
"""

from ..utils.colab_env import *
from ..storage.drive_manager import setup_storage_structure

def initialize_training_environment(base_dir: str = "/content/drive/MyDrive/TranningLorasData"):
    return setup_storage_structure(base_dir)

def install_all_trainer_dependencies(quiet: bool = True):
    return install_colab_prerequisites(quiet)
