import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import tempfile

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="YOLOv8 Detection", layout="wide")
st.title("📷 YOLOv8 Object Detection (Deployment Ready)")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # auto-download

model = load_model()

# -----------------------------
# SELECT MODE
# -----------------------------
option = st.radio("Choose Input Type:", ["Image", "Video"])

# -----------------------------
# IMAGE DETECTION
# -----------------------------
if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        results = model(image)
        result_img = results[0].plot()

        st.image(result_img, caption="Detected Objects", use_container_width=True)

# -----------------------------
# VIDEO DETECTION
# -----------------------------
elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated_frame = results[0].plot()

            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

        cap.release()
