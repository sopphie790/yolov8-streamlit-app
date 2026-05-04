import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import av
import cv2

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Vision SaaS Pro | Tracking System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# UI
# =========================
st.markdown("""
<style>
.main { background-color: #0e1117; }
.block-container { padding-top: 2rem; }
.dashboard-title { font-size: 28px; font-weight: 700; color: white; }
.subtitle { color: #a0a0a0; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL (FIXED)
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()  # 🔴 FIXED: GLOBAL MODEL

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("⚙️ Control Panel")
    mode = st.selectbox("Select Mode", ["Live Camera", "Image Upload"])
    conf = st.slider("Confidence Threshold", 0.1, 1.0, 0.25)

# =========================
# HEADER
# =========================
st.markdown("<div class='dashboard-title'>🎯 AI Vision Pro Dashboard</div>", unsafe_allow_html=True)

# =========================
# DETECTION FUNCTION (FIXED)
# =========================
def detect_frame(frame):
    results = model.predict(frame, conf=conf, verbose=False)

    annotated = results[0].plot()
    count = len(results[0].boxes) if results[0].boxes is not None else 0

    return annotated, count

# =========================
# TRACKING FUNCTION
# =========================
def track_frame(frame):
    results = model.track(frame, conf=conf, persist=True, verbose=False)

    annotated = results[0].plot()
    count = len(results[0].boxes) if results[0].boxes is not None else 0

    return annotated, count

# =========================
# MAIN APP
# =========================

if mode == "Live Camera":
    st.subheader("📷 Live Camera Detection")

    img_file = st.camera_input("Capture Frame")

    if img_file:
        image = Image.open(img_file)
        image = np.array(image)

        processed, count = detect_frame(image)  # FIXED FUNCTION

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original")

        with col2:
            st.image(processed, caption=f"Detected Objects: {count}")

elif mode == "Image Upload":
    st.subheader("🖼️ Object Tracking")

    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded:
        image = Image.open(uploaded)
        image = np.array(image)

        processed, count = track_frame(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original")

        with col2:
            st.image(processed, caption=f"Tracked Objects: {count}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("<center style='color:gray'>YOLOv8 SaaS Pro | Fixed Stable Version</center>", unsafe_allow_html=True)