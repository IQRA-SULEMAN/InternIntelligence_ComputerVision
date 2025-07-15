import streamlit as st
import torch
from PIL import Image
import os

st.set_page_config(page_title="YOLOv5 Object Detector", layout="centered")

path = "yolov5/runs/train/aquarium_model2/weights/best.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path=path, force_reload=True)

st.title("Aquarium Object Detection")
st.markdown("Upload an image to detect fish, jellyfish, and more!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting..."):
        results = model(image)
        results.render() 
        st.image(results.ims[0], caption="Detected Image", use_column_width=True)
