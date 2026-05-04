import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="YOLOv8 Detection", layout="wide")

# =========================
# Sidebar
# =========================
st.sidebar.title("⚙️ Settings")
mode = st.sidebar.selectbox(
    "Choose Mode",
    ["📷 Camera Capture", "🖼 Upload Image"]
)

confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.info("Developed using YOLOv8 + Streamlit")

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =========================
# Title
# =========================
st.title("🚀 YOLOv8 Object Detection App")
st.write("Detect objects using AI (Camera or Upload)")

# =========================
# CAMERA MODE
# =========================
if mode == "📷 Camera Capture":
    img_file = st.camera_input("Take a picture")

    if img_file is not None:
        image = Image.open(img_file)
        img = np.array(image)

        results = model(img, conf=confidence)
        annotated = results[0].plot()

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Original", use_column_width=True)

        with col2:
            st.image(annotated, caption="Detected", use_column_width=True)

        # Count objects
        counts = {}
        for cls in results[0].boxes.cls:
            label = model.names[int(cls)]
            counts[label] = counts.get(label, 0) + 1

        st.subheader("📊 Object Count")
        st.json(counts)

# =========================
# UPLOAD MODE
# =========================
elif mode == "🖼 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img = np.array(image)

        results = model(img, conf=confidence)
        annotated = results[0].plot()

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Original", use_column_width=True)

        with col2:
            st.image(annotated, caption="Detected", use_column_width=True)

        # Count objects
        counts = {}
        for cls in results[0].boxes.cls:
            label = model.names[int(cls)]
            counts[label] = counts.get(label, 0) + 1

        st.subheader("📊 Object Count")
        st.json(counts)