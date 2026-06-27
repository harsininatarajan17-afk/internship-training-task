import streamlit as st
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image
import numpy as np

st.header("72. Object Detection")

uploaded_file = st.file_uploader("Upload Image",type=["jpg","jpeg","png"],key="object_detection")

if uploaded_file:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, width=300)

    model = MobileNetV2(weights="imagenet")
    img_resized = img.resize((224,224))

    arr = image.img_to_array(img_resized)

    arr = np.expand_dims(arr, axis=0)

    arr = preprocess_input(arr)

    pred = model.predict(arr)

    results = decode_predictions(pred, top=5)[0]

    st.subheader("Detected Objects")

    for _, label, score in results:
        st.write(f"{label} : {score*100:.2f}%")