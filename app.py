import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Vision Dashboard | Live Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS (PRO UI)
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
    .dashboard-title {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
    }
    .subtitle {
        color: #a0a0a0;
        font-size: 14px;
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =========================
# SIDEBAR UI
# =========================
with st.sidebar:
    st.title("⚙️ Control Panel")
    mode = st.selectbox("Select Mode", ["Live Camera", "Image Upload"])
    conf = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)
    st.markdown("---")
    st.info("AI-powered real-time object detection system")

# =========================
# HEADER
# =========================
st.markdown("<div class='dashboard-title'>🎯 Live Object Detection & Tracking Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Professional AI Vision System powered by YOLOv8</div>", unsafe_allow_html=True)

# =========================
# METRICS ROW
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "YOLOv8", "Active")
with col2:
    st.metric("Status", "Live Ready", "🟢")
with col3:
    st.metric("Mode", mode, "⚡")

st.markdown("---")

# =========================
# DETECTION FUNCTIONS
# =========================
def detect_frame(frame):
    results = model.predict(frame, conf=conf)
    annotated = results[0].plot()
    return annotated, len(results[0].boxes)

# =========================
# MAIN APP
# =========================
if mode == "Live Camera":
    st.subheader("📷 Live Camera Feed")

    run = st.checkbox("Start Camera")

    if run:
        cap = cv2.VideoCapture(0)

        frame_placeholder = st.empty()
        stat_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not accessible")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed, count = detect_frame(frame)

            frame_placeholder.image(processed, channels="RGB")
            stat_placeholder.markdown(f"### Detected Objects: {count}")

            time.sleep(0.03)

        cap.release()

elif mode == "Image Upload":
    st.subheader("🖼️ Upload Image")

    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded:
        image = Image.open(uploaded)
        image = np.array(image)

        processed, count = detect_frame(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original")
        with col2:
            st.image(processed, caption=f"Detected Objects: {count}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("<center style='color:gray'>Built with Streamlit + YOLOv8 | AI Vision Dashboard</center>", unsafe_allow_html=True)
