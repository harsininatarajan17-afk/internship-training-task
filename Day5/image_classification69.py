import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import numpy as np
from PIL import Image

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

# Load pretrained model
model = MobileNetV2(weights="imagenet")

# Title
st.title("Image Classification using MobileNetV2")

# Upload image
uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image and convert RGBA -> RGB
    img = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.image(img, caption="Uploaded Image", width=300)

    # Resize image
    img = img.resize((224, 224))

    # Convert to numpy array
    img_array = np.array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess image
    img_array = preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array)

    # Decode predictions
    results = decode_predictions(predictions, top=3)[0]

    st.subheader("Top Predictions")

    for _, label, score in results:
        st.write(f"✅ {label}: {score:.2%}")