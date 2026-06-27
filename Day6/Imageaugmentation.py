import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
st.title("Image Augmentation")
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Original Image")
    col1, col2 = st.columns(2)
    with col1:
        rotated = img.rotate(30)
        st.image(rotated, caption="Rotated")
        flipped = ImageOps.mirror(img)
        st.image(flipped, caption="Flipped")
    with col2:
        zoomed = img.resize(
            (int(img.width*1.5), int(img.height*1.5))
        )
        st.image(zoomed, caption="Zoomed")
        gray = ImageOps.grayscale(img)
        st.image(gray, caption="Grayscale")