import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import av
import cv2

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Live Object Detection & Tracking",
    layout="wide"
)

st.title("🎥 Live Object Detection & Tracking")
st.write("Objects are detected and tracked in real-time using YOLOv8.")

# -----------------------------
# LOAD MODEL (CACHED)
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -----------------------------
# VIDEO PROCESSOR CLASS
# (IMPORTANT FOR STREAMLIT DEPLOYMENT)
# -----------------------------
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO tracking
        results = self.model.track(
            source=img,
            persist=True,
            conf=0.5,
            verbose=False
        )

        # Annotate frame
        annotated_frame = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# -----------------------------
# WEBCAM STREAM
# -----------------------------
webrtc_streamer(
    key="yolo-live",
    video_processor_factory=YOLOVideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    async_processing=True,
)
