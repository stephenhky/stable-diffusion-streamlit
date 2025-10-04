
import PIL
from typing import Union, Literal, Optional
import os
from os import PathLike
from pathlib import Path
import warnings

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer
from compel import Compel

from .schemas import StableDiffusionOutput


def get_stable_diffusion_pipeline(
        base_model_id: Literal["runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-xl-base-1.0"],
        lora_weights_path: str,
        vae_input_path: Union[PathLike, str],
        cuda: bool = True,     # quite impractical without CUDA
        text_encoder_id: Optional[str] = None,
        tokenizer_id: Optional[str] = None
) -> StableDiffusionPipeline:
    if text_encoder_id is not None and tokenizer_id is not None:
        text_encoder = AutoModel.from_pretrained(text_encoder_id)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            safety_checker=None,  # Disable for freedom
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            safety_checker=None,  # Disable for freedom
        )
        if (text_encoder_id is not None) ^ (tokenizer_id is not None):
            warnings.warn("You must give `text_encoder_id` and `tokenizer_id` a value or both None! Now it is assumed both are None!")
    if cuda and torch.cuda.is_available():
        pipe = pipe.to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if (vae_input_path is not None) and (len(vae_input_path if isinstance(vae_input_path, str) else vae_input_path.as_posix()) > 0):
        vae_weights = load_file(vae_input_path)
        pipe.vae.load_state_dict(vae_weights, strict=False)
    if (lora_weights_path is not None) and (len(lora_weights_path) > 0):
        pipe.load_lora_weights(lora_weights_path, weight_name="pytorch_lora_weights.safetensors")
    return pipe


def generate_image(
        positive_prompt: str,
        negative_prompt: str,
        stable_diffusion_pipeline: StableDiffusionPipeline,
        height: int = 512,
        width: int = 512,
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
) -> list[PIL.Image]:
    if seed is None:
        generator = None
    else:
        generator = torch.Generator("cuda").manual_seed(seed)

    # conditioning embedding
    compel = Compel(tokenizer=stable_diffusion_pipeline.tokenizer, text_encoder=stable_diffusion_pipeline.text_encoder)
    positive_prompt_embeds = compel(positive_prompt)
    negative_prompt_embeds = compel(negative_prompt)

    sd_output = stable_diffusion_pipeline(
        # prompt=positive_prompt,
        # negative_prompt=negative_prompt,
        prompt_embeds=positive_prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="pil"
    )

    return sd_output.images


class SDImageGenerator:
    def __init__(
            self,
            base_model_id: Literal["runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-xl-base-1.0"],
            lora_weight_path: Optional[Union[PathLike, str]],
            vae_weight_path: Optional[Union[PathLike, str]],
            text_encoder_id: Optional[str] = None,
            tokenizer_id: Optional[str] = None,
            cuda: bool=True
    ):
        self._base_model_id = base_model_id
        if isinstance(lora_weight_path, Path):
            self._lora_file_name = lora_weight_path.name
            lora_weight_path = lora_weight_path.as_posix()
        elif isinstance(lora_weight_path, str):
            self._lora_file_name = os.path.basename(lora_weight_path)
        else:
            self._lora_file_name = None
        if isinstance(vae_weight_path, str):
            vae_weight_path = Path(vae_weight_path)
            self._vae_file_name = vae_weight_path.name
        elif isinstance(vae_weight_path, Path):
            self._vae_file_name = vae_weight_path.name
        else:
            self._vae_file_name = None

        self._pipe = get_stable_diffusion_pipeline(
            self._base_model_id,
            lora_weight_path,
            vae_weight_path,
            text_encoder_id=text_encoder_id,
            tokenizer_id=tokenizer_id,
            cuda=cuda
        )

    def generate_images(
            self,
            positive_prompt: str,
            negative_prompt: str,
            height: int = 512,
            width: int = 512,
            steps: int = 30,
            guidance_scale: float = 7.5,
            seed: Optional[int] = None,
            nbimages: int = 1
    ) -> StableDiffusionOutput:
        images = [
            generate_image(
              positive_prompt,
              negative_prompt,
              self._pipe,
              height=height,
              width=width,
              steps=steps,
              guidance_scale=guidance_scale,
              seed=seed
          )[0]
          for _ in range(nbimages)
        ]
        return StableDiffusionOutput(
            base_model_id=self._base_model_id,
            lora_file_name=self._lora_file_name,
            vae_file_name=self._vae_file_name,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            images=images
        )
