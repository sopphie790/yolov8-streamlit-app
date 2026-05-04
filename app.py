import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Enterprise AI CCTV System",
    page_icon="📡",
    layout="wide"
)

# =========================
# UI DESIGN (CCTV STYLE)
# =========================
st.markdown("""
<style>
.main {
    background-color: #0b0f1a;
}

/* SIDEBAR CCTV STYLE */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        rgba(0, 0, 0, 0.85),
        rgba(20, 20, 20, 0.85)
    );
    backdrop-filter: blur(10px);
}

/* TEXT */
h1, h2, h3 {
    color: #00ffcc;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(90deg, #00ffcc, #00b3ff);
    color: black;
    border-radius: 10px;
    font-weight: bold;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# =========================
# MODEL (TRACKING ENABLED)
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

CONF = 0.25

# =========================
# SESSION STATE (CCTV LOGS)
# =========================
if "logs" not in st.session_state:
    st.session_state.logs = []

# =========================
# CCTV DETECTION (WITH TRACKING)
# =========================
def process_frame(frame):
    results = model.track(frame, persist=True, conf=CONF, verbose=False)

    annotated = results[0].plot()

    boxes = results[0].boxes

    class_counts = {}
    detected = []

    if boxes is not None:
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            name = model.names[cls_id]
            detected.append(name)

            if name in class_counts:
                class_counts[name] += 1
            else:
                class_counts[name] = 1

    return annotated, class_counts, detected

# =========================
# ANALYTICS PANEL
# =========================
def analytics_panel(class_counts):
    if not class_counts:
        return

    df = pd.DataFrame(list(class_counts.items()), columns=["Object", "Count"])

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Top Object", df.sort_values("Count", ascending=False).iloc[0]["Object"])

    with col2:
        st.metric("Total Types", len(df))

    fig, ax = plt.subplots()
    ax.bar(df["Object"], df["Count"])
    ax.set_title("Live CCTV Detection Analytics")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45)

    st.pyplot(fig)

# =========================
# TITLE
# =========================
st.title("📡 Enterprise AI CCTV Surveillance System")
st.write("Real-time object tracking + analytics + monitoring dashboard")

# =========================
# SIDEBAR (CONTROL ROOM)
# =========================
with st.sidebar:
    st.header("🎛 CCTV Control Room")

    start = st.checkbox("▶ Start Surveillance")

    st.markdown("---")
    st.subheader("📊 Live Logs")

    if st.session_state.logs:
        for log in reversed(st.session_state.logs[-6:]):
            st.success(log)
    else:
        st.info("No activity detected")

# =========================
# MAIN STREAM
# =========================
if start:

    st.subheader("📷 LIVE CCTV FEED")

    frame_window = st.empty()
    analytics_window = st.empty()

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            st.error("Camera not accessible")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        processed, class_counts, detected = process_frame(frame)

        # LOG EVENTS
        if detected:
            log = f"{datetime.now().strftime('%H:%M:%S')} - {', '.join(set(detected))}"
            st.session_state.logs.append(log)

        frame_window.image(processed, channels="RGB")

        with analytics_window.container():
            analytics_panel(class_counts)

    cap.release()

else:
    st.info("Activate CCTV surveillance from sidebar")