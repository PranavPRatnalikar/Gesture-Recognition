# Gesture Recognition with MediaPipe

This project is a gesture recognition system built using the **MediaPipe** framework. It allows you to recognize hand gestures from different input sources such as your webcam, images, and videos. Additionally, it provides a **Streamlit** interface to make the system more interactive.

## Project Overview

The system is capable of recognizing gestures in real-time from:

- **Webcam**: Use your webcam to capture hand gestures.
- **Images**: Upload images for gesture recognition.
- **Videos**: Process video files to detect hand gestures.
- **Streamlit Interface**: A simple user interface using Streamlit to interact with the system.

The project relies on **MediaPipe**, a machine learning framework used for building perception pipelines like gesture detection.

## Setup Guide

# To keep the dependencies isolated, it's recommended to create a virtual environment:

bash
# For Windows
python -m venv venv

# For Linux/Mac
python3 -m venv venv

# Activate the virtual environment:
# For Windows
venv\Scripts\activate

# For Linux/Mac
source venv/bin/activate


# Once the virtual environment is activated, install the required Python packages:
pip install -r requirements.txt

# To use your webcam for gesture recognition, run:
python app.py

# To upload and recognize gestures in an image, run:
app2.py
 
# For processing a video file and detecting gestures in it, run:
app3.py

# For a user-friendly interface, you can launch the Streamlit app using the following command:
streamlit run appInterface.py


Hope this helps! If you have any questions, feel free to ask.

This `README.md` file provides a simple overview of the project, setup instructions, and usage guidance. It includes instructions for running each of the four Python files along with details about creating a virtual environment and installing dependencies. Let me know if you need further customization!
