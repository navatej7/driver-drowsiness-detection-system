# driver-drowsiness-detection-system

Real-time eye-blink monitoring system using OpenCV and dlib to detect driver drowsiness and trigger an alarm for safety.

🚦 Driver Drowsiness Detection with Computer Vision

This project is a real-time driver drowsiness detection system that leverages computer vision and facial landmark recognition to identify whether a driver is active, drowsy, or sleeping. If prolonged eye closure is detected, the system plays an alarm sound to help prevent accidents caused by fatigue.

🔍 How It Works

Face & Eye Detection

Uses dlib’s frontal face detector and a 68-point facial landmark predictor to locate and track the driver’s eyes in each video frame.

Eye Aspect Ratio (EAR) Calculation

Computes the ratio between vertical and horizontal distances of eye landmarks.

Based on the EAR, the system determines whether the eyes are open, partially closed, or fully closed.

State Classification

Active 🙂 → Eyes open, driver alert.

Drowsy 😴 → Eyes partially closed for several frames.

Sleeping 💤 → Eyes fully closed for several frames.

Audio Alert

When the driver is detected as sleeping, a looping alarm sound is triggered using pygame until the eyes reopen.

🛠️ Tech Stack

Python – Core programming language.

OpenCV – Real-time video capture and processing.

dlib – Face detection & 68-point landmark predictor.

imutils – Helper functions for easier landmark handling.

NumPy – Distance calculations for EAR.

pygame – Alarm sound playback.

🚀 Setup & Usage

Install the dependencies:

pip install opencv-python dlib imutils pygame numpy

Download shape_predictor_68_face_landmarks.dat from dlib’s official model repo .

Place an alarm audio file (e.g., music.wav) in your project folder.

Run the script:

python drowsiness_detection.py

Press ESC to stop the program.

📌 Features

Real-time monitoring through webcam.

Detects active, drowsy, and sleeping states.

Plays an alarm sound when eyes are closed beyond safe limits.

Visual feedback with bounding boxes, landmarks, and state labels.

🌍 Applications

🚗 Driver assistance systems to prevent fatigue-related accidents.

🏭 Workplace monitoring in safety-critical jobs.

🛌 Sleep research for tracking fatigue in controlled environments.
