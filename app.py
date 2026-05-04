import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Vision Alert System",
    page_icon="🚨",
    layout="wide"
)

# =========================
# UI DESIGN
# =========================
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        rgba(255, 77, 166, 0.75),
        rgba(255, 26, 117, 0.75)
    );
    backdrop-filter: blur(14px);
}

[data-testid="stSidebar"] * {
    color: white !important;
    font-weight: 600;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(90deg, #ff4da6, #ff1a75);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.6em 1em;
    font-weight: bold;
    width: 100%;
}

h1, h2, h3 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =========================
# MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =========================
# ALERT SYSTEM STATE
# =========================
if "alerts" not in st.session_state:
    st.session_state.alerts = []

# =========================
# TITLE
# =========================
st.title("🚨 AI Object Detection Alert System")
st.write("Real-time detection with automated AI alerts.")

# =========================
# ALERT FUNCTION
# =========================
def check_alerts(detected_classes):
    alert_keywords = ["person", "car", "truck", "knife", "bottle"]

    triggered = []

    for obj in detected_classes:
        if obj in alert_keywords:
            triggered.append(obj)

    if triggered:
        alert_msg = f"🚨 ALERT: {', '.join(triggered)} detected!"
        st.session_state.alerts.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": alert_msg
        })
        return alert_msg

    return None

# =========================
# DETECTION FUNCTION
# =========================
def detect(frame):
    results = model.predict(frame, conf=0.3, verbose=False)

    annotated = results[0].plot()

    boxes = results[0].boxes
    count = len(boxes) if boxes is not None else 0

    classes = []

    if boxes is not None:
        for c in boxes.cls:
            classes.append(model.names[int(c)])

    alert = check_alerts(classes)

    return annotated, count, classes, alert

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("⚙️ Control Panel")

    mode = st.selectbox("Select Mode", ["Live Camera", "Upload Image"])

    st.markdown("---")

    st.subheader("🚨 AI Alerts")

    if st.session_state.alerts:
        for a in reversed(st.session_state.alerts[-5:]):
            st.error(f"{a['time']} - {a['message']}")
    else:
        st.info("No alerts yet")

# =========================
# MAIN APP
# =========================

if mode == "Live Camera":
    st.subheader("📷 Live AI Detection + Alerts")

    img_file = st.camera_input("Open Camera")

    if img_file:
        image = Image.open(img_file).convert("RGB")
        image = np.array(image)

        processed, count, classes, alert = detect(image)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(image, caption="Original")

        with col2:
            st.image(processed, caption="AI Detection")

        with col3:
            st.metric("Objects", count)

        if alert:
            st.error(alert)

        st.write("Detected:", list(set(classes)))

elif mode == "Upload Image":
    st.subheader("🖼️ AI Detection + Alert System")

    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image = np.array(image)

        processed, count, classes, alert = detect(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original")

        with col2:
            st.image(processed, caption="Detected")

        st.metric("Objects Detected", count)

        if alert:
            st.error(alert)

        st.write("Detected Objects:", list(set(classes)))