import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Vision Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# LOAD YOLO MODEL (SAFE)
# =========================
@st.cache_resource
def load_model():
    from ultralytics import YOLO

    # IMPORTANT: force re-download safe weights
    model = YOLO("yolov8n.pt", task="detect")
    return model

# =========================
# UI DESIGN
# =========================
st.markdown("""
<h1 style='color:#38bdf8;'>🧠 AI Vision Dashboard</h1>
<p style='color:#aaa;'>YOLOv8 Object Detection (Streamlit Cloud Stable Version)</p>
<hr>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("⚙️ Control Panel")

    st.info("This version uses image upload (stable for deployment)")

    st.write("Model: YOLOv8n")
    st.write("Status: Ready 🚀")

# =========================
# DASHBOARD CARDS
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("AI Model", "YOLOv8")

with col2:
    st.metric("Mode", "Image Detection")

with col3:
    st.metric("Platform", "Streamlit Cloud")

st.write("---")

# =========================
# IMAGE DETECTION (SAFE REPLACEMENT FOR WEBCAM)
# =========================
st.subheader("📷 Upload Image for Detection")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    results = model.predict(img, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            label = f"{model.names[cls]} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    st.image(img, channels="BGR")

st.write("---")

st.caption("Developed by Liza Jaime | AI Vision System")