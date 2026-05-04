import streamlit as st

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Vision Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CLEAN PROFESSIONAL UI CSS
# =========================
st.markdown("""
<style>

/* App background */
.stApp {
    background-color: #0e1117;
    color: white;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Dashboard cards */
.card {
    background: #1f2937;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
    margin-bottom: 15px;
}

/* Title */
.title {
    font-size: 40px;
    font-weight: 800;
    color: #38bdf8;
}

/* Subtitle */
.subtitle {
    font-size: 16px;
    color: #9ca3af;
}

/* Sidebar text */
.css-1d391kg {
    color: white;
}

/* Buttons */
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #1d4ed8;
}

</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR CONTROL PANEL
# =========================
with st.sidebar:
    st.title("⚙️ AI Control Panel")

    st.markdown("### Detection Settings")

    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

    st.markdown("### Mode")
    mode = st.radio("Select Input", ["Live Camera (Demo)", "Image Upload (Future)"])

    st.write("---")

    st.info("This dashboard is ready for YOLOv8 integration 🚀")

# =========================
# MAIN HEADER
# =========================
st.markdown('<div class="title">🧠 AI Vision Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time object detection system powered by YOLOv8</div>', unsafe_allow_html=True)

st.write("---")

# =========================
# DASHBOARD CARDS
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <h3>🎯 AI Model</h3>
        <p>YOLOv8 Object Detection</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <h3>⚡ Performance</h3>
        <p>Optimized Streamlit UI</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <h3>📷 Input</h3>
        <p>Camera / Image Ready</p>
    </div>
    """, unsafe_allow_html=True)

st.write("---")

# =========================
# STATUS SECTION
# =========================
st.markdown("""
<div class="card">
    <h3>📊 System Status</h3>
    <p>✔ UI Loaded Successfully</p>
    <p>✔ Streamlit Cloud Ready</p>
    <p>✔ YOLO Integration Ready</p>
</div>
""", unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.write("---")
st.markdown("### 👩‍💻 Developed by Liza Jaime")
st.caption("AI Vision Dashboard | Streamlit + YOLOv8 Ready System")