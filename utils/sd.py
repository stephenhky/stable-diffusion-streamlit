
import os
from os import PathLike
from pathlib import Path
import re
from typing import Union, Literal, Optional
import warnings

import PIL
import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    UniPCMultistepScheduler,
)
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

from .schemas import StableDiffusionOutput

MODEL_ID_ALIASES = {
    "runwayml/stable-diffusion-v1-5": "stable-diffusion-v1-5/stable-diffusion-v1-5",
}

SCHEDULERS = {
    "Euler a": (EulerAncestralDiscreteScheduler, {}),
    "Euler": (EulerDiscreteScheduler, {}),
    "DPM++ 2M": (DPMSolverMultistepScheduler, {}),
    "DPM++ 2M Karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
    "DPM++ 2M SDE": (DPMSolverMultistepScheduler, {"algorithm_type": "sde-dpmsolver++"}),
    "DPM++ 2M SDE Karras": (
        DPMSolverMultistepScheduler,
        {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True},
    ),
    "DPM++ SDE": (DPMSolverSinglestepScheduler, {}),
    "DPM++ SDE Karras": (DPMSolverSinglestepScheduler, {"use_karras_sigmas": True}),
    "Heun": (HeunDiscreteScheduler, {}),
    "KDPM2": (KDPM2DiscreteScheduler, {}),
    "KDPM2 a": (KDPM2AncestralDiscreteScheduler, {}),
    "DDIM": (DDIMScheduler, {}),
    "UniPC": (UniPCMultistepScheduler, {}),
}


def _convert_a1111_segment_to_compel(segment: str) -> str:
    """Convert A1111 weighting syntax within a single prompt segment to Compel syntax."""
    # Remove inline A1111 <lora:name:weight> tags so they aren't tokenized as literal words
    segment = re.sub(r"<lora:[^>]+>", "", segment)

    # Convert [text:weight] -> (text)weight
    segment = re.sub(
        r"(?<!\\)\[([^\[\]]+?):\s*([0-9]*\.?[0-9]+)\s*(?<!\\)\]",
        r"(\1)\2",
        segment,
    )

    # Convert [text] (de-emphasis, nested supported) -> (text)-
    prev = None
    while prev != segment:
        prev = segment
        segment = re.sub(r"(?<!\\)\[([^\[\]]+?)(?<!\\)\]", r"(\1)-", segment)

    # Convert (text:weight) -> (text)weight (innermost first to handle nesting)
    prev = None
    while prev != segment:
        prev = segment
        segment = re.sub(
            r"(?<!\\)\(([^()]+?):\s*([0-9]*\.?[0-9]+)\s*(?<!\\)\)",
            r"(\1)\2",
            segment,
        )

    # In A1111, bare (text) multiplies weight by 1.1, whereas in Compel bare (text) is 1.0
    # unless followed by +, -, a weight number, or an operator (.and, .blend, .attend, .swap).
    out = []
    i = 0
    n = len(segment)
    while i < n:
        if segment[i] == ")" and (i == 0 or segment[i - 1] != "\\"):
            out.append(")")
            rest = segment[i + 1 :]
            if not re.match(r"^(?:[+\-]|[0-9]|\.(?:attend|blend|and|add|swap)\b)", rest):
                out.append("+")
            i += 1
        else:
            out.append(segment[i])
            i += 1
    return "".join(out)


def convert_a1111_to_compel(prompt: str) -> str:
    """Convert AUTOMATIC1111 prompt syntax (weights, brackets, BREAK) into Compel syntax."""
    if not prompt or not prompt.strip():
        return prompt
    # Leave prompts that already use Compel's top-level .and()/.blend() untouched
    if re.search(r"\)\s*\.(?:and|add|blend)\s*\(", prompt):
        return prompt

    chunks = re.split(r"\bBREAK\b", prompt)
    converted_chunks = [_convert_a1111_segment_to_compel(c).strip() for c in chunks]
    if len(converted_chunks) == 1:
        return converted_chunks[0]

    escaped_chunks = [c.replace('"', '\\"') for c in converted_chunks]
    joined = ", ".join(f'"{c}"' for c in escaped_chunks)
    return f"({joined}).and()"


def _is_sdxl_pipeline(pipe) -> bool:
    return (
        isinstance(pipe, StableDiffusionXLPipeline)
        or (getattr(pipe, "tokenizer_2", None) is not None and getattr(pipe, "text_encoder_2", None) is not None)
    )


def set_scheduler(pipe, scheduler_name: str = "Euler"):
    if scheduler_name not in SCHEDULERS:
        raise ValueError(f"Unknown scheduler '{scheduler_name}'. Available: {list(SCHEDULERS.keys())}")
    scheduler_cls, kwargs = SCHEDULERS[scheduler_name]
    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config, **kwargs)
    return pipe


def get_stable_diffusion_pipeline(
        base_model_id: Union[Literal["runwayml/stable-diffusion-v1-5", " stable-diffusion-v1-5/stable-diffusion-v1-5", "stabilityai/stable-diffusion-xl-base-1.0"], str],
        lora_weights_path: Optional[Union[PathLike, str]],
        vae_input_path: Optional[Union[PathLike, str]],
        cuda: bool = True,     # quite impractical without CUDA
        text_encoder_id: Optional[str] = None,
        tokenizer_id: Optional[str] = None,
        text_inversion_embedding_path: Optional[Union[PathLike, str]] = None,
        scheduler_name: str = "Euler",
) -> Union[StableDiffusionPipeline, StableDiffusionXLPipeline]:
    resolved_model_id = str(base_model_id).strip()
    if resolved_model_id in MODEL_ID_ALIASES and not os.path.exists(resolved_model_id):
        resolved_model_id = MODEL_ID_ALIASES[resolved_model_id]

    use_cuda = bool(cuda and torch.cuda.is_available())
    dtype = torch.float16 if use_cuda else torch.float32

    is_xl = "xl" in resolved_model_id.lower()
    pipeline_cls = StableDiffusionXLPipeline if is_xl else StableDiffusionPipeline

    extra_kwargs = {}
    if not is_xl:
        extra_kwargs["safety_checker"] = None  # Disable for freedom

    if text_encoder_id is not None and tokenizer_id is not None:
        text_encoder = AutoModel.from_pretrained(text_encoder_id, torch_dtype=dtype)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        extra_kwargs["text_encoder"] = text_encoder
        extra_kwargs["tokenizer"] = tokenizer
    elif (text_encoder_id is not None) ^ (tokenizer_id is not None):
        warnings.warn("You must give `text_encoder_id` and `tokenizer_id` a value or both None! Now it is assumed both are None!")

    if os.path.isfile(resolved_model_id) and resolved_model_id.endswith((".safetensors", ".ckpt")):
        pipe = pipeline_cls.from_single_file(
            resolved_model_id,
            torch_dtype=dtype,
            **extra_kwargs,
        )
    else:
        pipe = pipeline_cls.from_pretrained(
            resolved_model_id,
            torch_dtype=dtype,
            **extra_kwargs,
        )

    if use_cuda:
        pipe = pipe.to("cuda")

    set_scheduler(pipe, scheduler_name)

    if vae_input_path is not None:
        vae_path_str = vae_input_path if isinstance(vae_input_path, str) else vae_input_path.as_posix()
        if len(vae_path_str) > 0:
            if os.path.isfile(vae_path_str):
                try:
                    pipe.vae = AutoencoderKL.from_single_file(vae_path_str, torch_dtype=dtype)
                except Exception:
                    vae_weights = load_file(vae_path_str)
                    pipe.vae.load_state_dict(vae_weights, strict=False)
            else:
                pipe.vae = AutoencoderKL.from_pretrained(vae_path_str, torch_dtype=dtype)
            if use_cuda:
                pipe.vae = pipe.vae.to("cuda", dtype=dtype)

    if lora_weights_path is not None:
        lora_path_str = lora_weights_path if isinstance(lora_weights_path, str) else lora_weights_path.as_posix()
        if len(lora_path_str) > 0:
            if os.path.isfile(lora_path_str):
                pipe.load_lora_weights(
                    os.path.dirname(lora_path_str) or ".",
                    weight_name=os.path.basename(lora_path_str),
                )
            elif os.path.isdir(lora_path_str) and os.path.isfile(
                os.path.join(lora_path_str, "pytorch_lora_weights.safetensors")
            ):
                pipe.load_lora_weights(lora_path_str, weight_name="pytorch_lora_weights.safetensors")
            else:
                pipe.load_lora_weights(lora_path_str)

    if text_inversion_embedding_path is not None:
        pipe.load_textual_inversion(text_inversion_embedding_path, token='ng_deepnegative_v1_75t')
    return pipe


def generate_image(
        positive_prompt: Union[str, list[str]],
        negative_prompt: Union[str, list[str]],
        stable_diffusion_pipeline: Union[StableDiffusionPipeline, StableDiffusionXLPipeline],
        height: int = 512,
        width: int = 512,
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        clip_skip: int = 1,
        convert_a1111_syntax: bool = True,
) -> list[PIL.Image.Image]:
    if convert_a1111_syntax:
        if isinstance(positive_prompt, list):
            positive_prompt = [convert_a1111_to_compel(p) for p in positive_prompt]
        else:
            positive_prompt = convert_a1111_to_compel(positive_prompt)

        if isinstance(negative_prompt, list):
            negative_prompt = [convert_a1111_to_compel(p) for p in negative_prompt]
        else:
            negative_prompt = convert_a1111_to_compel(negative_prompt)

    batch_size = len(positive_prompt) if isinstance(positive_prompt, list) else 1
    if seed is None:
        generator = None
    elif batch_size > 1:
        generator = [torch.Generator("cpu").manual_seed(int(seed) + i) for i in range(batch_size)]
    else:
        generator = torch.Generator("cpu").manual_seed(int(seed))

    is_xl = _is_sdxl_pipeline(stable_diffusion_pipeline)

    if is_xl:
        compel = Compel(
            tokenizer=[stable_diffusion_pipeline.tokenizer, stable_diffusion_pipeline.tokenizer_2],
            text_encoder=[stable_diffusion_pipeline.text_encoder, stable_diffusion_pipeline.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            truncate_long_prompts=False,
        )
        positive_prompt_embeds, positive_pooled = compel(positive_prompt)
        negative_prompt_embeds, negative_pooled = compel(negative_prompt)
        [positive_prompt_embeds, negative_prompt_embeds] = compel.pad_conditioning_tensors_to_same_length(
            [positive_prompt_embeds, negative_prompt_embeds]
        )
        sd_output = stable_diffusion_pipeline(
            prompt_embeds=positive_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=positive_pooled,
            negative_pooled_prompt_embeds=negative_pooled,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
        )
    else:
        emb_type = (
            ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED
            if clip_skip and clip_skip >= 2
            else ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED
        )
        compel = Compel(
            tokenizer=stable_diffusion_pipeline.tokenizer,
            text_encoder=stable_diffusion_pipeline.text_encoder,
            returned_embeddings_type=emb_type,
            truncate_long_prompts=False,
        )
        positive_prompt_embeds = compel(positive_prompt)
        negative_prompt_embeds = compel(negative_prompt)
        [positive_prompt_embeds, negative_prompt_embeds] = compel.pad_conditioning_tensors_to_same_length(
            [positive_prompt_embeds, negative_prompt_embeds]
        )

        sd_output = stable_diffusion_pipeline(
            prompt_embeds=positive_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type="pil",
        )

    return sd_output.images


class SDImageGenerator:
    def __init__(
            self,
            base_model_id: Union[Literal["runwayml/stable-diffusion-v1-5", "stable-diffusion-v1-5/stable-diffusion-v1-5", "stabilityai/stable-diffusion-xl-base-1.0"], str],
            lora_weight_path: Optional[Union[PathLike, str]],
            vae_weight_path: Optional[Union[PathLike, str]],
            text_encoder_id: Optional[str] = None,
            tokenizer_id: Optional[str] = None,
            cuda: bool = True,
            text_inversion_embedding_path: Optional[Union[PathLike, str]] = None,
            scheduler_name: str = "Euler",
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
            cuda=cuda,
            text_inversion_embedding_path=text_inversion_embedding_path,
            scheduler_name=scheduler_name,
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
            nbimages: int = 1,
            clip_skip: int = 1,
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
                seed=(seed + i if seed is not None else None),
                clip_skip=clip_skip,
            )[0]
            for i in range(nbimages)
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
