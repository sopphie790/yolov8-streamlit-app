import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Object Detection Pro",
    page_icon="📸",
    layout="wide"
)

# =========================
# SAFE IMAGE PREPROCESSING
# =========================
def preprocess_image(image):
    img = np.array(image)

    # Fix RGBA issue (VERY IMPORTANT)
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Convert RGB → BGR for YOLO stability
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img

# =========================
# LOAD MODEL (CACHE)
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =========================
# SIDEBAR DASHBOARD
# =========================
with st.sidebar:
    st.title("⚙️ Control Panel")

    app_mode = st.radio(
        "Choose Mode",
        ["📷 Camera Detection", "🖼️ Image Upload"]
    )

    conf_threshold = st.slider(
        "Confidence Threshold",
        0.1, 1.0, 0.5, 0.05
    )

    st.markdown("---")
    st.info("YOLOv8 + Streamlit Cloud Ready 🚀")

# =========================
# MAIN HEADER
# =========================
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>
    📸 AI Object Detection Pro
    </h1>
    <p style='text-align: center;'>
    Stable YOLOv8 Deployment App (One File Version)
    </p>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([2, 1])

# =========================
# CAMERA MODE
# =========================
if app_mode == "📷 Camera Detection":
    with col1:
        img_file = st.camera_input("📷 Capture Image")

    if img_file is not None:
        start_time = time.time()

        image = Image.open(img_file).convert("RGB")
        img = preprocess_image(image)

        results = model.predict(img, conf=conf_threshold)
        annotated = results[0].plot()

        st.image(annotated, caption="Detected Objects", use_container_width=True)

        # =========================
        # OBJECT COUNTING
        # =========================
        counts = {}
        if results[0].boxes is not None:
            for cls in results[0].boxes.cls:
                label = model.names[int(cls)]
                counts[label] = counts.get(label, 0) + 1

        with col2:
            st.subheader("📊 Detection Stats")

            if counts:
                for k, v in counts.items():
                    st.metric(label=k, value=v)
            else:
                st.info("No objects detected")

        st.caption(f"⏱ Processing Time: {round(time.time() - start_time, 2)} sec")

# =========================
# IMAGE UPLOAD MODE
# =========================
elif app_mode == "🖼️ Image Upload":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        start_time = time.time()

        image = Image.open(uploaded_file).convert("RGB")
        img = preprocess_image(image)

        results = model.predict(img, conf=conf_threshold)
        annotated = results[0].plot()

        st.image(annotated, caption="Detected Objects", use_container_width=True)

        # =========================
        # OBJECT COUNTING
        # =========================
        counts = {}
        if results[0].boxes is not None:
            for cls in results[0].boxes.cls:
                label = model.names[int(cls)]
                counts[label] = counts.get(label, 0) + 1

        st.subheader("📊 Object Summary")
        st.json(counts)

        st.caption(f"⏱ Processing Time: {round(time.time() - start_time, 2)} sec")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<center>🚀 YOLOv8 + Streamlit | Stable Deployment Version</center>",
    unsafe_allow_html=True
)