import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av

# =========================
# PAGE CONFIG (PRO UI LOOK)
# =========================
st.set_page_config(
    page_title="AI Live Detection Dashboard",
    page_icon="🎥",
    layout="wide"
)

# =========================
# SIDEBAR DASHBOARD
# =========================
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("Configure your AI detection system")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.0, 1.0, 0.5, 0.05
)

start_camera = st.sidebar.toggle("🎥 Start Camera", value=True)

st.sidebar.markdown("---")
st.sidebar.info("YOLOv8 Live Detection System")

# =========================
# MAIN HEADER
# =========================
st.title("🎥 AI Object Detection Dashboard")
st.markdown("Real-time object detection using YOLOv8 + WebRTC")

if start_camera:
    st.success("🟢 Camera is running")
else:
    st.warning("🔴 Camera is stopped")

# =========================
# LOAD MODEL (CACHE)
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =========================
# VIDEO PROCESSING
# =========================
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    results = model.track(
        img,
        persist=True,
        conf=conf_threshold,
        verbose=False
    )

    annotated = results[0].plot()

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# =========================
# LIVE STREAM (WEBRTC)
# =========================
if start_camera:
    webrtc_streamer(
        key="pro-ai-dashboard",
        video_frame_callback=video_frame_callback,
        async_processing=True,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
    )
else:
    st.info("Camera is turned off from sidebar")