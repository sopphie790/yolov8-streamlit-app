import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import av
import cv2

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Live Object Detection", layout="wide")

st.title("🎥 Live Object Detection & Tracing")
st.write("Real-time AI detection using YOLOv8 + Webcam")

# =========================
# LOAD MODEL (CACHE)
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =========================
# GLOBAL COUNTER
# =========================
object_counter = {}

# =========================
# VIDEO PROCESSOR CLASS
# =========================
class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):
        global object_counter

        img = frame.to_ndarray(format="bgr24")

        # YOLO tracking
        results = model.track(
            img,
            persist=True,
            conf=0.5,
            verbose=False
        )

        annotated_frame = results[0].plot()

        # =========================
        # OBJECT COUNTING
        # =========================
        if results[0].boxes is not None:
            object_counter = {}
            for cls in results[0].boxes.cls:
                label = model.names[int(cls)]
                object_counter[label] = object_counter.get(label, 0) + 1

        # =========================
        # ALERT FEATURE
        # =========================
        if "cell phone" in object_counter:
            cv2.putText(
                annotated_frame,
                "📱 PHONE DETECTED!",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# =========================
# START STREAM
# =========================
webrtc_streamer(
    key="object-detection",
    video_processor_factory=VideoProcessor,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)

# =========================
# DISPLAY OBJECT COUNT
# =========================
st.subheader("📊 Detected Objects Count")
st.write(object_counter)