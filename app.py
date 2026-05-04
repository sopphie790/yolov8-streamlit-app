import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Vision Enterprise System",
    page_icon="🎥",
    layout="wide"
)

# =========================
# UI DESIGN (SAFE ADD-ON)
# =========================
st.markdown("""
<style>
.main { background-color: #0e1117; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(255,77,166,0.6), rgba(0,0,0,0.8));
    backdrop-filter: blur(10px);
}

.stButton>button {
    background: linear-gradient(90deg, #ff4da6, #ff1a75);
    color: white;
    border-radius: 10px;
    width: 100%;
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
CONF = 0.25

# =========================
# SESSION STATE (LOGS)
# =========================
if "logs" not in st.session_state:
    st.session_state.logs = []

# =========================
# ORIGINAL DETECTION FUNCTION (KEPT)
# =========================
def detect(frame):
    results = model.predict(frame, conf=CONF, verbose=False)

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

    total = sum(class_count.values())

    return annotated, total, classes, class_count

# =========================
# 🔥 NEW: CCTV TRACKING (ADDED ONLY)
# =========================
def cctv_process(frame):
    results = model.track(frame, persist=True, conf=CONF, verbose=False)

    annotated = results[0].plot()

    boxes = results[0].boxes
    detected = []

    if boxes is not None:
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            name = model.names[cls_id]
            detected.append(name)

    return annotated, detected

# =========================
# 🔥 NEW: ANALYTICS DASHBOARD
# =========================
def analytics(class_count):
    if not class_count:
        return

    df = pd.DataFrame(list(class_count.items()), columns=["Object", "Count"])

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Top Object", df.sort_values("Count", ascending=False).iloc[0]["Object"])

    with col2:
        st.metric("Types Detected", len(df))

    fig, ax = plt.subplots()
    ax.bar(df["Object"], df["Count"])
    ax.set_title("AI Detection Analytics")
    plt.xticks(rotation=45)

    st.pyplot(fig)

# =========================
# TITLE (YOUR ORIGINAL STYLE PRESERVED)
# =========================
st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# =========================
# SIDEBAR (SAFE EXPANSION ONLY)
# =========================
with st.sidebar:
    st.header("⚙️ Control Panel")

    mode = st.selectbox(
        "Select Mode",
        ["Live Camera (Original)", "Upload Image (Original)", "🔥 CCTV Enterprise Mode"]
    )

    st.markdown("---")

    st.subheader("📊 Logs")

    if st.session_state.logs:
        for log in reversed(st.session_state.logs[-5:]):
            st.write(log)
    else:
        st.info("No logs yet")

# =========================
# ORIGINAL LIVE CAMERA (UNCHANGED LOGIC)
# =========================
if mode == "Live Camera (Original)":

    img_file = st.camera_input("Open Camera")

    if img_file:
        image = Image.open(img_file).convert("RGB")
        image = np.array(image)

        processed, count, classes, class_count = detect(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original")

        with col2:
            st.image(processed, caption="Detection")

        st.metric("Objects", count)

        st.write("Detected:", list(set(classes)))
        st.write("Class Breakdown:", class_count)

        analytics(class_count)

# =========================
# ORIGINAL UPLOAD (FIXED BUT KEPT)
# =========================
elif mode == "Upload Image (Original)":

    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image = np.array(image)

        processed, count, classes, class_count = detect(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image)

        with col2:
            st.image(processed)

        st.metric("Objects Detected", count)

        st.write("Detected:", list(set(classes)))
        st.write("Class Breakdown:", class_count)

        analytics(class_count)

# =========================
# 🔥 CCTV ENTERPRISE MODE (NEW ADD-ON ONLY)
# =========================
elif mode == "🔥 CCTV Enterprise Mode":

    st.subheader("📡 Live CCTV Monitoring System")

    start = st.checkbox("Start Surveillance")

    frame_holder = st.empty()

    cap = cv2.VideoCapture(0)

    while start and cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        processed, detected = cctv_process(frame)

        frame_holder.image(processed)

        if detected:
            log = f"{datetime.now().strftime('%H:%M:%S')} - {', '.join(set(detected))}"
            st.session_state.logs.append(log)

    cap.release()