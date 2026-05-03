import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import os
from datetime import datetime

# Load YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# Create folder for saved images
SAVE_DIR = "detected_frames"
os.makedirs(SAVE_DIR, exist_ok=True)

# Settings
ALERT_CLASSES = ["person", "cell phone"]
SAVE_CLASSES = ["person"]
last_saved_time = 0

# Frame processing
def video_frame_callback(frame):
    global last_saved_time

    img = frame.to_ndarray(format="bgr24")

    # Run tracking
    results = model.track(
        img,
        persist=True,
        conf=0.5,
        verbose=False
    )

    annotated_frame = results[0].plot()

    # Get detections
    boxes = results[0].boxes
    names = model.names

    person_count = 0
    detected_classes = []

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = names[cls_id]
            detected_classes.append(class_name)

            # Count persons
            if class_name == "person":
                person_count += 1

    # Display count on frame
    cv2.putText(
        annotated_frame,
        f"People: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Alerts
    for cls in ALERT_CLASSES:
        if cls in detected_classes:
            cv2.putText(
                annotated_frame,
                f"ALERT: {cls.upper()} DETECTED!",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

    # Save frame (limit saving to avoid spam)
    current_time = datetime.now().timestamp()
    if any(cls in detected_classes for cls in SAVE_CLASSES):
        if current_time - last_saved_time > 3:  # save every 3 seconds
            filename = os.path.join(
                SAVE_DIR,
                f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )
            cv2.imwrite(filename, annotated_frame)
            last_saved_time = current_time

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# Start webcam
webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)
