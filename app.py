import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from datetime import datetime

# =========================
# PAGE CONFIG (ENTERPRISE SAAS)
# =========================
st.set_page_config(
    page_title="Enterprise AI Tracking SaaS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# GLOBAL STYLE
# =========================
st.markdown("""
<style>
.main { background-color: #0b0f1a; }
.block-container { padding-top: 2rem; }
.title { font-size: 30px; font-weight: 800; color: white; }
.sub { color: #9aa4b2; }
.card { background: rgba(255,255,255,0.06); padding: 15px; border-radius: 12px; }
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
# SESSION STATE (ENTERPRISE LOGGING)
# =========================
if "log_data" not in st.session_state:
    st.session_state.log_data = []

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("🧠 Enterprise Control Panel")
    mode = st.selectbox("Mode", ["Live Camera", "Image Analysis"])
    conf = st.slider("Confidence", 0.1, 1.0, 0.3)
    show_log = st.checkbox("Show Detection Logs")

# =========================
# HEADER
# =========================
st.markdown("<div class='title'>🧠 Enterprise AI Tracking SaaS</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>Real-time Object Intelligence + Analytics Engine</div>", unsafe_allow_html=True)

# =========================
# CORE FUNCTIONS
# =========================
def detect(frame):
    results = model.predict(frame, conf=conf, verbose=False)
    annotated = results[0].plot()
    boxes = results[0].boxes

    count = len(boxes) if boxes is not None else 0

    detected_classes = []
    if boxes is not None:
        for c in boxes.cls:
            detected_classes.append(model.names[int(c)])

    # LOGGING (ENTERPRISE FEATURE)
    st.session_state.log_data.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "count": count,
        "objects": ", ".join(detected_classes)
    })

    return annotated, count, detected_classes

# =========================
# MAIN APP
# =========================
if mode == "Live Camera":
    st.subheader("📷 Live Intelligence Feed")

    img_file = st.camera_input("Capture Frame")

    if img_file:
        image = Image.open(img_file)
        image = np.array(image)

        processed, count, classes = detect(image)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(image, caption="Original")

        with col2:
            st.image(processed, caption="AI Detection")

        with col3:
            st.metric("Objects Detected", count)

            st.write("Detected Classes:")
            st.write(list(set(classes)))

elif mode == "Image Analysis":
    st.subheader("🖼️ Enterprise Image Analysis")

    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded:
        image = Image.open(uploaded)
        image = np.array(image)

        processed, count, classes = detect(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original")

        with col2:
            st.image(processed, caption="AI Processed")

        st.success(f"Detected {count} objects")

        st.write("### Detected Objects")
        st.write(list(set(classes)))

# =========================
# ANALYTICS DASHBOARD
# =========================
if show_log:
    st.markdown("---")
    st.subheader("📊 Detection Analytics (Enterprise Log)")

    if st.session_state.log_data:
        df = pd.DataFrame(st.session_state.log_data)
        st.dataframe(df)

        st.bar_chart(df["count"])
    else:
        st.info("No logs yet")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("<center style='color:gray'>Enterprise AI SaaS | Tracking + Analytics Engine</center>", unsafe_allow_html=True)