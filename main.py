import time
import torch
import streamlit as st
from diffusers import StableDiffusionPipeline

st.set_page_config(layout="wide")
st.title("Stable Diffusion")

st.text("Loading Stable Diffusion Model")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

with st.form("prompt", clear_on_submit=False):
    prompt = st.text_area("Prompt: ", value="")
    submit_button = st.form_submit_button(label="Enter")
    if submit_button:
        image = pipe(prompt).images[0]

    if submit_button:
        col1, col2, col3 = st.columns(3)

        with col2:
            st.image(image)
