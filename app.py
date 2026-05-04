import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av

# =========================
# PAGE CONFIG (PRO LOOK)
# =========================
st.set_page_config(
    page_title="AI Detection Dashboard",
    page_icon="🎥",
    layout="wide"
)

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.title("⚙️ AI Control Panel")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

camera_on = st.sidebar.toggle("🎥 Start Live Camera", value=True)

st.sidebar.markdown("---")
st.sidebar.success("YOLOv8 + WebRTC System Ready")

# =========================
# MAIN DASHBOARD UI
# =========================
st.title("🎥 AI Object Detection Dashboard")
st.markdown("Real-time detection using YOLOv8 + Streamlit WebRTC")

if camera_on:
    st.success("🟢 Live Camera Active")
else:
    st.warning("🔴 Camera Stopped")

# =========================
# LOAD MODEL (SAFE CACHE)
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

    annotated_frame = results[0].plot()

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# =========================
# WEBSRTC STREAM (DEPLOY SAFE)
# =========================
if camera_on:
    webrtc_streamer(
        key="ai-live-detection",
        video_frame_callback=video_frame_callback,
        async_processing=True,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )
else:
    st.info("Camera is turned off. Enable it from sidebar.")