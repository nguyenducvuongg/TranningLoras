from setuptools import setup, find_packages

setup(
    name="lora_trainer",
    version="1.0.0",
    description="Universal and Optimized LoRA Training Toolkit for Google Colab (Image & Video)",
    author="Nguyen Duc Vuong",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "safetensors>=0.4.2",
        "toml>=0.10.2",
        "pyyaml>=6.0",
        "pillow>=10.0.0",
        "requests>=2.31.0",
        "tqdm>=4.66.0",
    ],
)
