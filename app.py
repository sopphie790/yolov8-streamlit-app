import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Object Detection App", layout="centered")

st.title("📸 AI Object Detection (Camera Capture)")
st.write("Capture an image and detect objects using YOLOv8")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =========================
# CAMERA INPUT
# =========================
img_file = st.camera_input("📷 Take a picture")

if img_file is not None:
    start_time = time.time()

    image = Image.open(img_file)
    img = np.array(image)

    # =========================
    # YOLO DETECTION
    # =========================
    results = model(img)

    annotated_frame = results[0].plot()

    st.image(annotated_frame, caption="Detected Objects", use_column_width=True)

    # =========================
    # OBJECT COUNTING
    # =========================
    counts = {}
    if results[0].boxes is not None:
        for cls in results[0].boxes.cls:
            label = model.names[int(cls)]
            counts[label] = counts.get(label, 0) + 1

    st.subheader("📊 Object Count")
    st.write(counts)

    # =========================
    # ALERT FEATURE
    # =========================
    if "cell phone" in counts:
        st.warning("📱 Cellphone detected!")

    if "person" in counts:
        st.success("👤 Person detected!")

    # =========================
    # PERFORMANCE INFO
    # =========================
    end_time = time.time()
    st.caption(f"⏱ Detection Time: {round(end_time - start_time, 2)} seconds")