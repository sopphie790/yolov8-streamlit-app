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
# PROFESSIONAL UI (PINK SIDEBAR FIXED)
# =========================
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}

/* FIXED SIDEBAR (TRANSPARENT + MODERN) */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(255,77,166,0.95), rgba(255,26,117,0.95));
    backdrop-filter: blur(12px);
}

/* SIDEBAR TEXT */
[data-testid="stSidebar"] * {
    color: white !important;
    font-weight: 500;
}

/* BUTTON STYLE FIXED */
.stButton>button {
    background: linear-gradient(90deg, #ff4da6, #ff1a75);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.6em 1em;
    font-weight: bold;
    transition: 0.3s;
    width: 100%;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 4px 20px rgba(255, 26, 117, 0.4);
}

/* TITLE STYLE */
h1, h2, h3 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SAFE MODEL LOADING (FIX FOR CLOUD ERRORS)
# =========================
@st.cache_resource
def load_model():
    try:
        model = YOLO("yolov8n.pt")
        return model
    except Exception as e:
        st.error("Model loading failed. Check requirements.txt or Torch install.")
        return None

model = load_model()

# =========================
# TITLE (YOUR REQUEST)
# =========================
st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# =========================
# DETECTION FUNCTION
# =========================
def detect(frame):
    if model is None:
        return frame, 0

    results = model.predict(frame, conf=0.3, verbose=False)

    annotated = results[0].plot()
    count = len(results[0].boxes) if results[0].boxes is not None else 0

    return annotated, count

# =========================
# SIDEBAR
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
    st.subheader("📷 Live Camera Detection")

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
            st.image(image, caption="Original")

        with col2:
            st.image(processed, caption=f"Detected Objects: {count}")