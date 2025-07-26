
from PIL import Image

from pydantic import BaseModel, ConfigDict


class StableDiffusionOutput(BaseModel):
    model_config = ConfigDict(from_attributes=True)

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
    images: list[Image]
