"""
AI Captioning Base System & Task Presets
Định nghĩa các kịch bản Prompt chuyên biệt cho từng dạng bài toán LoRA (Nhân vật, Skin/Portrait, Phục hồi, Phong cách, Tổng quát).
"""

from typing import Dict, Optional

TASK_PROMPT_PRESETS: Dict[str, str] = {
    "General": (
        "Describe the image in a concise, precise, and descriptive natural language paragraph. "
        "Focus on the main subject, background, lighting, artistic style, and key elements."
    ),
    "Face_Likeness": (
        "Describe the clothing, hairstyle, hair color, pose, gaze direction, camera perspective, lighting, and background environment in detail. "
        "Crucially: Do NOT describe the person's unique facial features, eye shape, nose, mouth, lips, or facial contours, "
        "ensuring the LoRA trigger word captures 100% of the individual's facial likeness and biometric identity without leakage."
    ),
    "Product_Commercial": (
        "Describe the commercial scene, studio lighting reflections, pedestal or surface material, composition, background environment, and product category. "
        "Do NOT describe the proprietary shape, unique logo, trademark emblems, or specific brand markings in generic terms, "
        "allowing the trigger word to capture 100% of the authentic product geometry and branding."
    ),
    "Skin_Portrait": (
        "Focus strictly on high-detail facial texture, skin micro-details, blemishes, pores, natural wrinkles, "
        "subsurface scattering, realistic lighting on the face, eye reflections, and authentic non-plastic human skin features. "
        "Avoid over-smoothing or airbrushed descriptions."
    ),
    "Character_Outfit": (
        "Describe the pose, clothing, outfit details, fabric texture, accessories, background environment, and camera angle. "
        "Do not describe the person's specific facial features or facial identity, so the model learns to associate the identity with the trigger word."
    ),
    "Art_Style": (
        "Describe the subject, colors, and composition clearly, but completely omit mentioning the art style, medium, or artist name. "
        "The model must learn the style purely through the trigger word."
    ),
    "Upscale_Restoration": (
        "Describe the fine photographic details, crisp edges, sharpness, depth of field, realistic materials, "
        "reflections, and high-fidelity rendering aspects of the image."
    ),
}

LENGTH_CONSTRAINTS: Dict[str, str] = {
    "Short": "Keep the caption under 25 words.",
    "Medium": "Keep the caption around 40-70 words.",
    "Long": "Provide a comprehensive, detailed description of 80-120 words.",
}


def build_task_prompt(
    task_mode: str = "General",
    caption_length: str = "Medium",
    trigger_word: Optional[str] = None,
) -> str:
    """Tạo câu lệnh Prompt hoàn chỉnh cho Vision LLM theo yêu cầu bài toán."""
    base_instr = TASK_PROMPT_PRESETS.get(task_mode, TASK_PROMPT_PRESETS["General"])
    len_instr = LENGTH_CONSTRAINTS.get(caption_length, LENGTH_CONSTRAINTS["Medium"])

    prompt = f"{base_instr} {len_instr}"
    if trigger_word and trigger_word.strip():
        prompt += f" Always start the description with the trigger word '{trigger_word.strip()}':"

    return prompt
