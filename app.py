import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Vision SaaS Pro",
    page_icon="🎥",
    layout="wide"
)

# =========================
# PROFESSIONAL UI (PINK SIDEBAR)
# =========================
st.markdown("""
<style>
/* MAIN BACKGROUND */
.main {
    background-color: #0e1117;
}

/* SIDEBAR PINK DESIGN */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ff4da6, #ff1a75);
}

/* SIDEBAR TEXT */
[data-testid="stSidebar"] * {
    color: white !important;
    font-weight: 500;
}

/* BUTTON STYLE */
.stButton>button {
    background: linear-gradient(90deg, #ff4da6, #ff1a75);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6em 1em;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #ff1a75, #ff4da6);
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
# TITLE (AS REQUESTED)
# =========================
st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# =========================
# DETECTION FUNCTION
# =========================
def detect(frame):
    results = model.predict(frame, conf=0.3)
    annotated = results[0].plot()
    count = len(results[0].boxes) if results[0].boxes is not None else 0
    return annotated, count

# =========================
# SIDEBAR MENU
# =========================
with st.sidebar:
    st.header("⚙️ Control Panel")
    mode = st.selectbox("Select Mode", ["Live Camera", "Upload Image"])
    st.markdown("---")
    st.info("AI SaaS Object Detection System")

# =========================
# MAIN APP
# =========================

if mode == "Live Camera":
    st.subheader("📷 Camera Detection")

    img_file = st.camera_input("Open Camera")

    if img_file:
        image = Image.open(img_file)
        image = np.array(image)

        processed, count = detect(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original")

        with col2:
            st.image(processed, caption=f"Detected Objects: {count}")

elif mode == "Upload Image":
    st.subheader("🖼️ Image Detection")

    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded:
        image = Image.open(uploaded)
        image = np.array(image)

        processed, count = detect(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image)

        with col2:
            st.image(processed, caption=f"Detected Objects: {count}")