import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Object Detection Pro",
    page_icon="📸",
    layout="wide"
)

# =========================
# SIDEBAR DESIGN
# =========================
with st.sidebar:
    st.title("📌 Navigation Panel")
    st.markdown("### AI Detection System")
    st.write("Select options below:")

    app_mode = st.radio(
        "Choose Mode",
        ["📷 Camera Detection", "🖼️ Image Upload"]
    )

    st.markdown("---")
    st.info("YOLOv8 AI Model running on Streamlit Cloud ready 🚀")

# =========================
# HEADER DESIGN
# =========================
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>
    📸 AI Object Detection Pro
    </h1>
    <p style='text-align: center;'>
    Real-time detection using YOLOv8 (Streamlit Deployment Ready)
    </p>
    """,
    unsafe_allow_html=True
)

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =========================
# MAIN APP
# =========================
col1, col2 = st.columns([2, 1])

# -------------------------
# CAMERA MODE
# -------------------------
if app_mode == "📷 Camera Detection":
    with col1:
        img_file = st.camera_input("📷 Capture Image")

    if img_file is not None:
        start_time = time.time()

        image = Image.open(img_file)
        img = np.array(image)

        results = model(img)
        annotated_frame = results[0].plot()

        st.image(annotated_frame, caption="Detected Objects", use_container_width=True)

        # =========================
        # OBJECT COUNTING
        # =========================
        counts = {}
        if results[0].boxes is not None:
            for cls in results[0].boxes.cls:
                label = model.names[int(cls)]
                counts[label] = counts.get(label, 0) + 1

        with col2:
            st.subheader("📊 Detection Stats")

            if counts:
                for k, v in counts.items():
                    st.metric(label=k, value=v)
            else:
                st.info("No objects detected")

            # Alerts
            if "person" in counts:
                st.success("👤 Person detected")
            if "cell phone" in counts:
                st.warning("📱 Cellphone detected")

        end_time = time.time()
        st.caption(f"⏱ Processing Time: {round(end_time - start_time, 2)} sec")

# -------------------------
# IMAGE UPLOAD MODE
# -------------------------
elif app_mode == "🖼️ Image Upload":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        start_time = time.time()

        image = Image.open(uploaded_file)
        img = np.array(image)

        results = model(img)
        annotated_frame = results[0].plot()

        st.image(annotated_frame, caption="Detected Objects", use_container_width=True)

        counts = {}
        if results[0].boxes is not None:
            for cls in results[0].boxes.cls:
                label = model.names[int(cls)]
                counts[label] = counts.get(label, 0) + 1

        st.subheader("📊 Object Summary")
        st.json(counts)

        end_time = time.time()
        st.caption(f"⏱ Processing Time: {round(end_time - start_time, 2)} sec")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<center>🚀 Built with Streamlit + YOLOv8 | Enhanced UI Version</center>",
    unsafe_allow_html=True
)