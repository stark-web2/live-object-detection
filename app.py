import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="Live Object Detection", layout="wide")
st.title("🎥 Live Object Detection & Tracking (YOLOv8)")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -----------------------------
# SIDEBAR OPTIONS
# -----------------------------
st.sidebar.header("Settings")

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.3, 1.0, 0.5)
alert_object = st.sidebar.text_input("Alert Object (e.g. bottle, person)", "person")

# -----------------------------
# CAMERA INPUT
# -----------------------------
img_file_buffer = st.camera_input("Take a picture for detection")

if img_file_buffer is not None:
    # Convert image
    image = Image.open(img_file_buffer)
    img_array = np.array(image)

    # Run YOLO detection
    results = model.predict(img_array, conf=conf_threshold)

    annotated_frame = results[0].plot()

    # -----------------------------
    # OBJECT COUNTING
    # -----------------------------
    boxes = results[0].boxes
    names = model.names

    person_count = 0
    alert_triggered = False

    for box in boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]

        if label == "person":
            person_count += 1

        if label == alert_object:
            alert_triggered = True

    # -----------------------------
    # DISPLAY RESULTS
    # -----------------------------
    st.image(annotated_frame, caption="Detected Objects", use_container_width=True)

    st.write(f"👥 People Count: {person_count}")

    # -----------------------------
    # ALERT SYSTEM
    # -----------------------------
    if alert_triggered:
        st.error(f"⚠ ALERT: {alert_object} detected!")
    else:
        st.success("No alert objects detected")
