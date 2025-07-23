
import PIL
from typing import Union, Literal
from os import PathLike

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from safetensors.torch import load_file


def get_stable_diffusion_pipeline(
        base_model_id: Literal["runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-xl-base-1.0"],
        lora_weights_path: str,
        vae_input_path: Union[PathLike, str],
        cuda: bool = True
) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        safety_checker=None,  # Disable for freedom
    )
    if cuda and torch.cuda.is_available():
        pipe = pipe.to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if (vae_input_path is not None) and (len(vae_input_path) > 0):
        vae_weights = load_file(vae_input_path)
        pipe.vae.load_state_dict(vae_weights, strict=False)
    if (lora_weights_path is not None) and (len(lora_weights_path) > 0):
        pipe.load_lora_weights(lora_weights_path, weight_name="pytorch_lora_weights.safetensors")
    return pipe


def generate_image(
        positive_prompt: str,
        negative_prompt: str,
        stable_diffusion_pipeline: StableDiffusionPipeline,
        height: int=512,
        width: int=512,
        steps: int=30,
        guidance_scale: float=7.5,
        seed: int=None
) -> list[PIL.Image]:
    if seed is None:
        generator = None
    else:
        generator = torch.Generator("cuda").manual_seed(seed)

    sd_output = stable_diffusion_pipeline(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="pil"
    )

    return sd_output.images
