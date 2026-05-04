import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import av
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Vision SaaS Pro | Tracking System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM UI
# =========================
st.markdown("""
<style>
.main { background-color: #0e1117; }
.block-container { padding-top: 2rem; }
.dashboard-title { font-size: 28px; font-weight: 700; color: white; }
.subtitle { color: #a0a0a0; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL (TRACKING ENABLED)
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")

model = load_model()

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("⚙️ Control Panel")
    mode = st.selectbox("Select Mode", ["Live Camera (WebRTC)", "Image Upload"])
    conf = st.slider("Confidence Threshold", 0.1, 1.0, 0.25)
    st.info("Multi-Object Tracking System (YOLOv8 + ByteTrack)")

# =========================
# HEADER
# =========================
st.markdown("<div class='dashboard-title'>🎯 AI Vision Pro Tracking Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Real-time Multi-Object Detection + Tracking (ID enabled)</div>", unsafe_allow_html=True)

# =========================
# TRACKING FUNCTION
# =========================
def track_frame(frame):
    results = model.track(frame, conf=conf, persist=True, verbose=False)

    annotated = results[0].plot()

    count = 0
    if results[0].boxes is not None:
        count = len(results[0].boxes)

        # Draw tracking IDs manually (extra clarity)
        for box in results[0].boxes:
            if box.id is not None:
                x1, y1, x2, y2 = box.xyxy[0]
                track_id = int(box.id[0])
                cv2.putText(
                    annotated,
                    f"ID {track_id}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

    return annotated, count

# =========================
# WEBRTC CONFIG
# =========================
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# =========================
# VIDEO TRANSFORMER (TRACKING)
# =========================
class YOLOTracker(VideoTransformerBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        processed, count = track_frame(img)

        cv2.putText(
            processed,
            f"Tracked Objects: {count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return av.VideoFrame.from_ndarray(processed, format="bgr24")

# =========================
# MAIN APP
# =========================
if mode == "Live Camera (WebRTC)":
    st.subheader("📷 Live Detection (Stable Mode)")

    img_file = st.camera_input("Capture Frame")

    if img_file:
        image = Image.open(img_file)
        image = np.array(image)

        processed, count = detect_frame(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original")

        with col2:
            st.image(processed, caption=f"Detected Objects: {count}")

elif mode == "Image Upload":
    st.subheader("🖼️ Object Tracking (Image)")

    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded:
        image = Image.open(uploaded)
        image = np.array(image)

        processed, count = track_frame(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original")

        with col2:
            st.image(processed, caption=f"Tracked Objects: {count}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("<center style='color:gray'>YOLOv8 Multi-Object Tracking SaaS Pro | ByteTrack Enabled</center>", unsafe_allow_html=True)