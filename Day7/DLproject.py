import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing.image import img_to_array

st.set_page_config(page_title="Transfer Learning", page_icon="🧠")

st.title("🧠 85. Transfer Learning")
st.write("Upload an image to classify it.")

@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((224, 224))

    img_array = img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    with st.spinner("Analyzing..."):

        prediction = model.predict(img_array)

    results = decode_predictions(prediction, top=5)[0]

    st.success("Prediction Completed")

    st.subheader("Top 5 Predictions")

    for i, (_, label, score) in enumerate(results, start=1):
        st.write(f"{i}. **{label.replace('_',' ').title()}** : {score*100:.2f}%")

    st.subheader("Confidence")

    for _, label, score in results:
        st.progress(float(score))
        st.write(f"{label}: {score*100:.2f}%")