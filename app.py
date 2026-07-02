import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import joblib
import streamlit as st
deep_model = load_model("brain_tumor_mlp.keras")
encoder = joblib.load("label_encoder.pkl")
st.set_page_config(page_title="Brain Tumor MRI Classifier")
st.title("Brain Tumor MRI Classification")
st.write("Upload brain MRI image to classify")
st.sidebar.write("""This application classifies brain MRI images using:
- Random Forest
- Multi-Layer Perceptron (MLP)

Dataset: Brain Tumor MRI Dataset""")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

from PIL import Image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption= "Uploaded MRI Image", use_container_width=True)

if uploaded_file is not None and st.button("Predict"):


    img = image.convert("L")

   
    img = img.resize((64, 64))

    
    img = np.array(img)

    
    img = img / 255.0

    
    img = img.reshape(1, 4096)

    prediction = deep_model.predict(img)

    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = encoder.inverse_transform([predicted_class])[0]

    st.success(f"Predicted Tumor Type: {predicted_label}")