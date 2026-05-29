# ==========================================
# Pneumonia Detection Web App (Streamlit)
# ==========================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ------------------------------------------
# Page Configuration
# ------------------------------------------

st.set_page_config(
    page_title="Pneumonia Detection AI",
    layout="centered"
)

st.title("🩺 Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image to detect Pneumonia")

# ------------------------------------------
# Load Trained Model
# ------------------------------------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("pneumonia_model.h5")
    return model

model = load_model()

# ------------------------------------------
# Image Preprocessing Function
# ------------------------------------------

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ------------------------------------------
# File Upload
# ------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

# ------------------------------------------
# Prediction
# ------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    if st.button("Predict"):

        processed_image = preprocess_image(image)

        prediction = model.predict(processed_image)[0][0]

        confidence = float(prediction)

        st.subheader("Prediction Result")

        if prediction > 0.5:
            st.error("⚠️ Pneumonia Detected")
            st.write(f"Confidence: {confidence:.2f}")
        else:
            st.success("✅ Normal Chest X-ray")
            st.write(f"Confidence: {1-confidence:.2f}")