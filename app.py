import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

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

/* SIDEBAR */
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

.stButton>button:hover {
    color: black !important;
    transform: scale(1.03);
}

h1, h2, h3 {
    color: white;
}
</style>
""", unsafe_allow_html=True)

/* =========================
   CONTROL ROOM UI PRO SIDEBAR
========================= */

/* Sidebar base glass effect */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        rgba(10, 10, 10, 0.95),
        rgba(20, 20, 20, 0.90)
    );
    backdrop-filter: blur(14px);
    border-right: 1px solid rgba(255, 77, 166, 0.3);
}

/* Sidebar header */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ff4da6 !important;
    letter-spacing: 1px;
}

/* Control panel title glow */
[data-testid="stSidebar"] h2 {
    text-shadow: 0 0 10px rgba(255, 77, 166, 0.5);
}

/* Selectbox (mode selector) PRO LOOK */
[data-testid="stSelectbox"] > div {
    background: rgba(255, 255, 255, 0.06);
    border: 2px solid rgba(255, 77, 166, 0.8);
    border-radius: 12px;
    padding: 6px;
    transition: 0.2s ease-in-out;
}

/* Select text */
[data-testid="stSelectbox"] * {
    color: white !important;
    font-weight: 600;
}

/* Arrow visibility fix */
[data-testid="stSelectbox"] svg {
    fill: #ffffff !important;
    opacity: 1 !important;
}

/* Hover glow */
[data-testid="stSelectbox"]:hover {
    box-shadow: 0 0 15px rgba(255, 77, 166, 0.6);
    border-color: #ff1a75;
}

/* Sidebar info / alerts box */
[data-testid="stSidebar"] .stAlert {
    border-radius: 10px;
    background: rgba(255, 26, 117, 0.1);
    border-left: 3px solid #ff4da6;
}

/* Divider styling */
[data-testid="stSidebar"] hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #ff4da6, transparent);
    margin: 15px 0;
}

/* Sidebar text readability */
[data-testid="stSidebar"] * {
    color: #f5f5f5 !important;
    font-weight: 500;
}

/* Smooth transition for UI */
[data-testid="stSidebar"] * {
    transition: all 0.2s ease-in-out;
}

# =========================
# MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

CONF_THRESHOLD = 0.2

# =========================
# ALERT SYSTEM
# =========================
if "alerts" not in st.session_state:
    st.session_state.alerts = []

if "last_alert" not in st.session_state:
    st.session_state.last_alert = ""

def check_alerts(detected_classes):
    alert_keywords = ["person", "car", "truck", "knife", "bottle"]

    triggered = [obj for obj in detected_classes if obj in alert_keywords]

    if triggered:
        alert_msg = f"🚨 ALERT: {', '.join(set(triggered))} detected!"

        if alert_msg != st.session_state.last_alert:
            st.session_state.alerts.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "message": alert_msg
            })
            st.session_state.last_alert = alert_msg

        return alert_msg

    return None

# =========================
# DETECTION FUNCTION
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

            if name in class_count:
                class_count[name] += 1
            else:
                class_count[name] = 1

    alert = check_alerts(classes)

    total_count = sum(class_count.values())

    return annotated, total_count, classes, alert, class_count

# =========================
# 🔥 AI ANALYTICS DASHBOARD
# =========================
def render_analytics(class_count):
    if not class_count:
        return

    df = pd.DataFrame(list(class_count.items()), columns=["Object", "Count"])

    st.subheader("📊 AI Analytics Dashboard")

    top = df.sort_values("Count", ascending=False).iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Top Object", top["Object"])

    with col2:
        st.metric("Highest Count", int(top["Count"]))

    fig, ax = plt.subplots()
    ax.bar(df["Object"], df["Count"])
    ax.set_title("Detection Breakdown")
    ax.set_xlabel("Objects")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)

    st.pyplot(fig)

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

        processed, count, classes, alert, class_count = detect(image)

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
        st.write("Class Breakdown:", class_count)

        render_analytics(class_count)

elif mode == "Upload Image":
    st.subheader("🖼️ AI Detection + Alert System")

    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image = np.array(image)

        processed, count, classes, alert, class_count = detect(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original")

        with col2:
            st.image(processed, caption="Detected")

        st.metric("Objects Detected", count)

        if alert:
            st.error(alert)

        st.write("Detected Objects:", list(set(classes)))
        st.write("Class Breakdown:", class_count)

        render_analytics(class_count)