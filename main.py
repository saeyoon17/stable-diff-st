import torch
import streamlit as st
from diffusers import StableDiffusionPipeline

VESSL_LOGO_URL = (
    "https://vessl-public-apne2.s3.ap-northeast-2.amazonaws.com/vessl-logo/vessl-ai_color_light"
    "-background.png"
)

st.set_page_config(layout="wide")
st.image(VESSL_LOGO_URL, width=500)
st.title("VESSL AI: Manage your own Stable Diffusion session!")
st.text("Setting environment is one of the biggest bottleneck for machine learning projects. 😥")
st.text("VESSL AI lets you overcome the bottleneck with the use of yaml. 📋")
st.text(
    "By providing yaml, you can declaratively run your machine learning projects in reliable manner!"
)
st.text("Try your own Stable Diffusion session with simple yaml we provide. 🚀")

# Load model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

with st.form("prompt", clear_on_submit=False):
    prompt = st.text_area("Write down your prompt here!: ", value="")
    submit_button = st.form_submit_button(label="Enter")
    if submit_button:
        image = pipe(prompt).images[0]

    if submit_button:
        col1, col2, col3 = st.columns(3)

        with col2:
            st.image(image)

col4, col5 = st.columns(2)
with col4:
    st.text("Here, we provide the yaml we used for setting up this streamlit session.")
    yaml = """
name : stable-diffusion
resources:
  cluster: aws-apne2-prod1
  accelerators: Tesla-V100:1
image: nvcr.io/nvidia/pytorch:21.05-py3
run:
  - workdir: /root/stable-diff-st/
    command: |
      bash ./setup.sh
volumes:
  /root/stable-diff-st: git://github.com/saeyoon17/stable-diff-st
interactive:
  runtime: 24h
  ports:
    - 8501
    """
    st.markdown(yaml)

with col5:
    st.text("You can save the yaml and run it by userself! Try:")
    st.markdown("pip install vessl")
    st.markdown("vessl run -f youryaml")
    st.text("to get your first yaml started!")

st.text("For further details, visit somelink.com!")
