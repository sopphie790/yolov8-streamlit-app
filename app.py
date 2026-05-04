import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

st.title("🎥 Live Object Detection (Simulated)")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Failed to access camera")
        break

    # YOLO detection
    results = model(frame)
    annotated_frame = results[0].plot()

    FRAME_WINDOW.image(annotated_frame, channels="BGR")

camera.release()