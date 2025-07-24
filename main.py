
import streamlit as st

from utils.sd import get_stable_diffusion_pipeline, generate_image

base_model = st.radio(
    "Base model",
    ["runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-xl-base-1.0"]
)
lora_weights_path = st.text_input("LoRA weights path")
vae_input_path = st.text_input("VAE path")
cuda = st.checkbox("CUDA", False)

height = st.number_input("Height", min_value=1, value=512)
width = st.number_input("Width", min_value=1, value=512)
nbsteps = st.number_input("Steps", min_value=1, value=30)
guidance_scale = st.number_input("Guidance scale", min_value=0.0, value=7.5)
seed = st.number_input("Seed", min_value=-1, value=-1)

nbimages = st.number_input("Number of images", min_value=1, value=1)

positive_prompt = st.text_area("Positive prompt")
negative_prompt = st.text_area("Negative prompt")

if st.button("Generate image(s)!"):
    if seed == -1:
        seed = None
    if len(lora_weights_path) == 0:
        lora_weights_path = None
    if len(vae_input_path) == 0:
        vae_input_path = None

    pipe = get_stable_diffusion_pipeline(
        base_model,
        lora_weights_path,
        vae_input_path,
        cuda=cuda
    )

    images = generate_image(
        [positive_prompt]*nbimages,
        [negative_prompt]*nbimages,
        pipe,
        height=height,
        width=width,
        steps=nbsteps,
        guidance_scale=guidance_scale,
        seed=seed
    )

    for image in images:
        st.image(image)
