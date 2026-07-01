import streamlit as st
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

if st.button("Predict"):
    st.success("Prediction will appear here after model is added.")
