import streamlit as st
from ultralytics import YOLO
import cv2
import os
import tempfile
import subprocess
from datetime import datetime
from imageio_ffmpeg import get_ffmpeg_exe

st.title("Dhaka Traffic Detection Using YOLOv11")
st.write("Upload a Traffic video or choose a sample video to detect objects:")

@st.cache_resource
def load_model():
    model = YOLO('./runs/detect/train/weights/best.pt')
    return model

model = load_model()

# Initialize session state
if "video_path" not in st.session_state:
    st.session_state.video_path = None
    st.session_state.processed_video_path = None

# Dropdown for selecting a sample video
sample_videos = {
    "Sample Traffic Video-1": "./inputs/input-1.mp4",
    "Sample Traffic Video-2": "./inputs/input-2.mp4",
}

selected_sample_video = st.radio("Choose a sample video:", ["None"] + list(sample_videos.keys()))
st.write("Please upload Traffic video under 30mb")
uploaded_file = st.file_uploader("Upload a Traffic Video..", type=["mp4", "avi"])

# Determine the video path based on user input
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        st.session_state.video_path = temp_file.name
    st.session_state.processed_video_path = None  # Reset previous output
elif selected_sample_video != "None":
    st.session_state.video_path = sample_videos[selected_sample_video]
    st.session_state.processed_video_path = None  # Reset previous output

# Display video preview
if st.session_state.video_path:
    st.write("Preview of selected video:")
    st.video(st.session_state.video_path)

# Process video if available
if st.session_state.video_path and st.button("Process Video"):
    st.write("Processing video...")
    cap = cv2.VideoCapture(st.session_state.video_path)

    # Generate a unique name for the output file using a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    yolo_output_path = f"./streamlit_output/yolo_output_{timestamp}.mp4"
    ffmpeg_output_path = f"./streamlit_output/ffmpeg_output_{timestamp}.mp4"

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define video writer
    os.makedirs("./streamlit_output", exist_ok=True)
    out = cv2.VideoWriter(yolo_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

    st.write("Post-processing video...")

    # FFmpeg processing
    ffmpeg_path = get_ffmpeg_exe()
    command = [
        ffmpeg_path, '-i', yolo_output_path,
        '-vcodec', 'libx264', '-acodec', 'aac',
        '-strict', 'experimental', ffmpeg_output_path
    ]

    subprocess.run(command)

    # Check if the file exists
    if os.path.exists(ffmpeg_output_path):
        st.session_state.processed_video_path = ffmpeg_output_path
        st.video(ffmpeg_output_path)
        st.success("Object detection complete! Here's a preview of your processed video.")
    else:
        st.error("Processed video file not found. Please check the file path.")
