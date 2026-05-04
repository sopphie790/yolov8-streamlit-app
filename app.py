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
    page_title="AI Vision SaaS Pro | Live Detection",
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
    .dashboard-title {
        font-size: 28px;
        font-weight: 700;
        color: white;
    }
    .subtitle {
        color: #a0a0a0;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("⚙️ Control Panel")
    mode = st.selectbox("Select Mode", ["Live Camera (WebRTC)", "Image Upload"])
    conf = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)
    st.info("AI SaaS Vision System")

# =========================
# HEADER
# =========================
st.markdown("<div class='dashboard-title'>🎯 AI Vision SaaS Pro Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Real-time Object Detection powered by YOLOv8</div>", unsafe_allow_html=True)

# =========================
# DETECTION FUNCTION
# =========================
def detect_frame(frame):
    results = model.predict(frame, conf=conf)
    annotated = results[0].plot()
    count = len(results[0].boxes)
    return annotated, count

# =========================
# WEBRTC CONFIG
# =========================
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# =========================
# VIDEO TRANSFORMER
# =========================
class YOLOTransformer(VideoTransformerBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        processed, count = detect_frame(img)

        cv2.putText(
            processed,
            f"Objects: {count}",
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
    st.subheader("📡 Live AI Detection (WebRTC Camera)")

    webrtc_streamer(
        key="ai-live",
        video_transformer_factory=YOLOTransformer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False}
    )

elif mode == "Image Upload":
    st.subheader("🖼️ Image Detection")

    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded:
        image = Image.open(uploaded)
        image = np.array(image)

        processed, count = detect_frame(image)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Original")

        with col2:
            st.image(processed, caption=f"Detected Objects: {count}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("<center style='color:gray'>AI SaaS Pro Dashboard | YOLOv8 + WebRTC</center>", unsafe_allow_html=True)