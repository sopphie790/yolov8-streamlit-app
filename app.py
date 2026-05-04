import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Vision Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# LOAD YOLO MODEL (cached)
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # lightweight model for cloud

model = load_model()

# =========================
# SAFE STUN CONFIG (IMPORTANT FOR STREAMLIT CLOUD CAMERA)
# =========================
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# =========================
# YOLO VIDEO PROCESSOR
# =========================
class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model(img, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                label = f"{model.names[cls]} {conf:.2f}"

                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(img, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0,255,0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# =========================
# UI DESIGN
# =========================
st.markdown("""
<h1 style='color:#38bdf8;'>🧠 AI Vision Dashboard</h1>
<p style='color:#aaa;'>Live YOLOv8 Object Detection (Streamlit Cloud Safe)</p>
<hr>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("⚙️ Control Panel")

    st.info("Live YOLO webcam uses streamlit-webrtc (cloud safe)")

    st.write("Model: YOLOv8n")
    st.write("Status: Ready 🚀")

# =========================
# DASHBOARD CARDS
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("AI Model", "YOLOv8")

with col2:
    st.metric("Mode", "Live Webcam")

with col3:
    st.metric("Platform", "Streamlit Cloud")

st.write("---")

# =========================
# LIVE CAMERA SECTION
# =========================
st.subheader("📷 Live Detection Camera")

webrtc_streamer(
    key="yolo-live",
    video_processor_factory=YOLOProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)

st.write("---")

st.caption("Developed by Liza Jaime | AI Vision System")