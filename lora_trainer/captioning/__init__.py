"""
AI Captioning Studio Subsystem for TranningLoras.
Hỗ trợ gán nhãn đa mô hình AI: Gemini 3.5/3.6/3.7, JoyCaption, Florence-2, OpenAI GPT-4o.
"""

from .base_captioner import TASK_PROMPT_PRESETS, LENGTH_CONSTRAINTS, build_task_prompt
from .gemini import batch_caption_gemini
from .joycaption import batch_caption_joy
from .florence import batch_caption_florence
from .openai_gpt import batch_caption_openai

__all__ = [
    "TASK_PROMPT_PRESETS",
    "LENGTH_CONSTRAINTS",
    "build_task_prompt",
    "batch_caption_gemini",
    "batch_caption_joy",
    "batch_caption_florence",
    "batch_caption_openai",
]
