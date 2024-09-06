import mediapipe as mp
import cv2

# Import the necessary modules from MediaPipe
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions, RunningMode
from mediapipe.tasks.python import vision

# Path to your gesture recognition model
model_path = r'D:/SIH 24/Women Safety/GestureRecognition1/hand_gesture_detection/MediaPipeGesture/gesture_recognizer.task'

# Callback function to handle gesture recognition results
def print_result(result, output_image, timestamp_ms):
    if result.gestures and len(result.gestures) > 0:
        gesture = result.gestures[0][0]  # Get the most confident gesture
        gesture_name = gesture.category_name
        print(f"Recognized Gesture: {gesture_name}")
    else:
        print("No gesture recognized.")

# Set up base options for the gesture recognizer
base_opts = base_options.BaseOptions(model_asset_path=model_path)

# Set up the gesture recognizer with live stream mode
options = GestureRecognizerOptions(
    base_options=base_opts,
    running_mode=RunningMode.LIVE_STREAM,
    result_callback=print_result  # Callback to handle the results
)

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

with GestureRecognizer.create_from_options(options) as recognizer:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the frame to RGB (since OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to a MediaPipe Image object
        mp_image_obj = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Get the current timestamp in milliseconds
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

        # Send the image to the recognizer for processing
        recognizer.recognize_async(mp_image_obj, timestamp_ms=timestamp_ms)

        # Display the image with the gesture result (optional)
        cv2.imshow('MediaPipe Gesture Recognition', frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release the webcam and close any open windows
cap.release()
cv2.destroyAllWindows()
