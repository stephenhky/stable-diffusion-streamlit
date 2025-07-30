
from PIL import Image
from dataclasses import dataclass


@dataclass
class StableDiffusionOutput:
    base_model_id: str
    lora_file_name: str
    vae_file_name: str
    positive_prompt: str
    negative_prompt: str
    height: int
    width: int
    steps: int
    guidance_scale: float
    seed: int
    images: list[Image.Image]
