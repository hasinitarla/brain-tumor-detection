# app.py

import streamlit as st
from PIL import Image
from utils import load_model, preprocess_image, predict

st.set_page_config(page_title="Brain Tumor Detector", layout="centered")

st.title("🧠 Brain Tumor Detection from MRI")
st.write("Upload an MRI scan to check for brain tumors.")

# Upload image
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🩺 Predict"):
        with st.spinner("Analyzing..."):
            model = load_model()
            image_tensor = preprocess_image(image)
            label, confidence = predict(model, image_tensor)

            class_names = ['No Tumor', 'Tumor Detected']
            st.success(f"Prediction: **{class_names[label]}**")
            st.info(f"Confidence: **{confidence * 100:.2f}%**")
