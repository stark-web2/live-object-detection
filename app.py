     import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import av

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="YOLOv8 Detection", layout="wide")
st.title("🎥 Live Object Detection & Tracking")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -----------------------------
# SIDEBAR SETTINGS
# -----------------------------
conf = st.sidebar.slider("Confidence", 0.1, 1.0, 0.5)

# -----------------------------
# VIDEO PROCESSOR
# -----------------------------
class YOLOProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model.track(
            img,
            persist=True,
            conf=conf,
            verbose=False
        )

        annotated = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# -----------------------------
# WEBCAM STREAM (BROWSER BASED)
# -----------------------------
webrtc_streamer(
    key="yolo",
    video_processor_factory=YOLOProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)
