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
# UI DESIGN FIXED
# =========================
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}

/* SIDEBAR TRANSPARENT FIX */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        rgba(255, 77, 166, 0.65),
        rgba(255, 26, 117, 0.65)
    );
    backdrop-filter: blur(14px);
}

/* SIDEBAR TEXT */
[data-testid="stSidebar"] * {
    color: white !important;
    font-weight: 600;
}

/* BUTTON STYLE + BLACK TEXT ON HOVER */
.stButton>button {
    background: linear-gradient(90deg, #ff4da6, #ff1a75);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.6em 1em;
    font-weight: bold;
    width: 100%;
    transition: 0.3s;
}

.stButton>button:hover {
    color: black !important;
    transform: scale(1.03);
    box-shadow: 0px 4px 20px rgba(255, 26, 117, 0.3);
}

/* TITLES */
h1, h2, h3 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =========================
# MODEL (IMPROVED DETECTION)
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# LOWER CONFIDENCE = MORE OBJECTS DETECTED
CONF_THRESHOLD = 0.2

# =========================
# ALERT STATE
# =========================
if "alerts" not in st.session_state:
    st.session_state.alerts = []

# prevent spam alerts
if "last_alert" not in st.session_state:
    st.session_state.last_alert = ""

# =========================
# TITLE
# =========================
st.title("🚨 AI Object Detection Alert System")
st.write("Real-time detection with smart AI alerts.")

# =========================
# ALERT SYSTEM
# =========================
def check_alerts(detected_classes):
    alert_keywords = ["person", "car", "truck", "knife", "bottle"]

    triggered = [obj for obj in detected_classes if obj in alert_keywords]

    if triggered:
        alert_msg = f"🚨 ALERT: {', '.join(set(triggered))} detected!"

        # prevent duplicate spam
        if alert_msg != st.session_state.last_alert:
            st.session_state.alerts.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "message": alert_msg
            })
            st.session_state.last_alert = alert_msg

        return alert_msg

    return None

# =========================
# DETECTION FUNCTION (FIXED)
# =========================
def detect(frame):
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)

    annotated = results[0].plot()

    boxes = results[0].boxes

    classes = []
    class_count = {}

    if boxes is not None:
        for c in boxes.cls:
            name = model.names[int(c)]
            classes.append(name)

            # accurate per-class counting
            if name in class_count:
                class_count[name] += 1
            else:
                class_count[name] = 1

    alert = check_alerts(classes)

    total_count = sum(class_count.values())

    return annotated, total_count, classes, alert, class_count

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

        st.write("Detected Objects:", list(set(classes)))

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