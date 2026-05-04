import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import time

# =========================
# PAGE CONFIG (SAAS STYLE)
# =========================
st.set_page_config(
    page_title="AI Vision SaaS Dashboard",
    page_icon="🤖",
    layout="wide"
)

# =========================
# GLOBAL STYLE (CLEAN DASHBOARD LOOK)
# =========================
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# IMAGE PREPROCESSING
# =========================
def preprocess_image(image):
    img = np.array(image)

    if img.ndim == 3 and img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =========================
# SIDEBAR (SAAS CONTROL PANEL)
# =========================
with st.sidebar:
    st.title("🤖 AI Control Center")

    st.markdown("### ⚙️ System Settings")

    app_mode = st.radio(
        "Mode",
        ["📷 Camera", "🖼️ Upload"]
    )

    conf_threshold = st.slider(
        "Confidence",
        0.1, 1.0, 0.5
    )

    st.markdown("---")
    st.success("System Online 🟢")

# =========================
# HEADER (SAAS HERO SECTION)
# =========================
st.markdown("""
    <h1 style='text-align:center; color:#00ffcc;'>
    🤖 AI Vision SaaS Dashboard
    </h1>
    <p style='text-align:center; color:gray;'>
    Real-time Object Detection Powered by YOLOv8
    </p>
""", unsafe_allow_html=True)

st.markdown("---")

# =========================
# LAYOUT
# =========================
col1, col2, col3 = st.columns([2, 1, 1])

# =========================
# INPUT SECTION
# =========================
if app_mode == "📷 Camera":
    img_file = st.camera_input("Capture Image")

else:
    img_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# =========================
# PROCESSING
# =========================
if img_file is not None:

    start_time = time.time()

    image = Image.open(img_file).convert("RGB")
    img = preprocess_image(image)

    results = model.predict(img, conf=conf_threshold)
    annotated = results[0].plot()

    # =========================
    # DETECTION RESULTS
    # =========================
    st.image(annotated, use_container_width=True)

    # =========================
    # ANALYTICS
    # =========================
    counts = {}
    total_objects = 0

    if results[0].boxes is not None:
        for cls in results[0].boxes.cls:
            label = model.names[int(cls)]
            counts[label] = counts.get(label, 0) + 1
            total_objects += 1

    processing_time = round(time.time() - start_time, 2)

    # =========================
    # DASHBOARD METRICS (SAAS STYLE)
    # =========================
    with col1:
        st.markdown("### 📊 Detection Summary")
        st.metric("Objects Detected", total_objects)

    with col2:
        st.markdown("### ⚡ Performance")
        st.metric("Processing Time", f"{processing_time}s")

    with col3:
        st.markdown("### 🎯 Model")
        st.metric("Confidence", f"{conf_threshold}")

    # =========================
    # DETAILED RESULTS
    # =========================
    st.markdown("---")
    st.subheader("📦 Detected Objects Breakdown")
    st.json(counts)

else:
    st.info("Upload or capture an image to start AI detection")