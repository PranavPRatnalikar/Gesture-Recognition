import streamlit as st
import mediapipe as mp
import cv2
import tempfile
import numpy as np
from PIL import Image

# Gesture Recognition Setup
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions, RunningMode

# Path to your gesture recognition model
model_path = r'D:/SIH 24/Women Safety/GestureRecognition1/hand_gesture_detection/MediaPipeGesture/gesture_recognizer.task'

# Function to recognize gesture in image
def recognize_gesture_in_image(image_file):
    img = Image.open(image_file)
    img_array = np.array(img)  # Convert image to a numpy array
    
    # Set up the gesture recognizer
    base_opts = base_options.BaseOptions(model_asset_path=model_path)
    options = GestureRecognizerOptions(
        base_options=base_opts,
        running_mode=RunningMode.IMAGE
    )

    # Convert the numpy array to a MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_array)
    
    with GestureRecognizer.create_from_options(options) as recognizer:
        # Perform gesture recognition
        gesture_recognition_result = recognizer.recognize(mp_image)
    
    if gesture_recognition_result.gestures and len(gesture_recognition_result.gestures) > 0:
        gesture = gesture_recognition_result.gestures[0][0]
        return f"Recognized Gesture: {gesture.category_name}"
    else:
        return "No gesture recognized."

# Function to recognize gesture in video
def recognize_gesture_in_video(video_file):
    # Create a temp file to save video for processing
    with tempfile.NamedTemporaryFile(delete=False) as tmp_video:
        tmp_video.write(video_file.read())
        video_path = tmp_video.name
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Setup Gesture Recognizer
    base_opts = base_options.BaseOptions(model_asset_path=model_path)
    options = GestureRecognizerOptions(
        base_options=base_opts,
        running_mode=RunningMode.VIDEO
    )

    with GestureRecognizer.create_from_options(options) as recognizer:
        frame_num = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((frame_num / fps) * 1000)
            
            result = recognizer.recognize_for_video(mp_image, timestamp_ms)
            if result.gestures and len(result.gestures) > 0:
                gesture = result.gestures[0][0]
                st.write(f"Frame: {frame_num}, Timestamp: {timestamp_ms}, Gesture: {gesture.category_name}")
            else:
                st.write(f"Frame: {frame_num}, No gesture recognized.")
            frame_num += 1
    cap.release()

# Function to recognize gesture from webcam
def recognize_gesture_from_webcam():
    cap = cv2.VideoCapture(0)
    
    base_opts = base_options.BaseOptions(model_asset_path=model_path)
    options = GestureRecognizerOptions(
        base_options=base_opts,
        running_mode=RunningMode.LIVE_STREAM,
        result_callback=print_result  # Set the callback function
    )
    
    with GestureRecognizer.create_from_options(options) as recognizer:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            recognizer.recognize_async(mp_image, timestamp_ms=timestamp_ms)
            
            cv2.imshow('Webcam', frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

# Callback function to handle the live stream result
def print_result(result, output_image, timestamp_ms):
    if result.gestures and len(result.gestures) > 0:
        gesture = result.gestures[0][0]
        st.write(f"Recognized Gesture: {gesture.category_name}")
    else:
        st.write("No gesture recognized.")

# Streamlit UI
st.title("Gesture Recognition App")

option = st.sidebar.selectbox(
    "Select Input Type",
    ("Upload Image", "Upload Video", "Open Webcam")
)

if option == "Upload Image":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        st.image(uploaded_image)
        result = recognize_gesture_in_image(uploaded_image)
        st.write(result)

elif option == "Upload Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        st.video(uploaded_video)
        recognize_gesture_in_video(uploaded_video)

elif option == "Open Webcam":
    st.write("Opening Webcam...")
    recognize_gesture_from_webcam()
