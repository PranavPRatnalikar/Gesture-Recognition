import mediapipe as mp
import cv2

# Initialize MediaPipe and OpenCV components
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Path to your gesture recognition model
model_path = r'gesture_recognizer.task'

# Setup the Gesture Recognizer with video mode
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

# Open the video file using OpenCV
video_path = r'5211959-uhd_2560_1440_25fps.mp4'
cap = cv2.VideoCapture(video_path)

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Open the Gesture Recognizer in a context manager
with GestureRecognizer.create_from_options(options) as recognizer:
    frame_num = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Convert the frame to RGB (since OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the frame to a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Calculate the timestamp in milliseconds
        timestamp_ms = int((frame_num / fps) * 1000)
        
        # Perform gesture recognition on the frame
        result = recognizer.recognize_for_video(mp_image, timestamp_ms)
        
        # Check if any gestures were recognized
        if result.gestures and len(result.gestures) > 0:
            gesture = result.gestures[0][0]  # Get the most confident gesture
            gesture_name = gesture.category_name
            print(f"Frame: {frame_num}, Time: {timestamp_ms}ms, Recognized Gesture: {gesture_name}")
            if(gesture_name=='Open_Palm'):
                print("ALERT")
        else:
            print(f"Frame: {frame_num}, Time: {timestamp_ms}ms, No gesture recognized.")
        
        frame_num += 1
    
    # Release the video capture object
    cap.release()

# Optionally, close all OpenCV windows
cv2.destroyAllWindows()
