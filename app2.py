import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions, RunningMode
from mediapipe.tasks.python.core import base_options  

model_path = r'gesture_recognizer.task'
image_path = r'hitler.jpeg'

base_options = base_options.BaseOptions(model_asset_path=model_path) 

# Create gesture recognizer options
options = GestureRecognizerOptions(
    base_options=base_options,
    running_mode=RunningMode.IMAGE
)

# Create the gesture recognizer from the options
with GestureRecognizer.create_from_options(options) as recognizer:
    # Load the input image from a file
    mp_image = mp.Image.create_from_file(image_path)
    
    # Perform gesture recognition
    gesture_recognition_result = recognizer.recognize(mp_image)

    # Extract and print the recognized gesture name
    if gesture_recognition_result.gestures and len(gesture_recognition_result.gestures) > 0:
        gesture = gesture_recognition_result.gestures[0][0]
        gesture_name = gesture.category_name
        print(f"Recognized Gesture: {gesture_name}")
    else:
        print("No gesture recognized.")
