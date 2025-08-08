import streamlit as st
import requests
from PIL import Image
import io

st.title("Smart Waste Classifier 🌍")

uploaded_file = st.file_uploader("Upload a waste image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://127.0.0.1:8000/predict/", files=files)
        
        if response.status_code == 200:
            prediction = response.json()['predicted_class']
            st.success(f"♻️ Predicted Waste Type: **{prediction}**")
        else:
            st.error("Prediction failed.")
