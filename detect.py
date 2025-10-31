## Import required libraries
import dlib                     # Used for face detection and landmark prediction
import numpy as np               # Used for numerical operations (arrays, vectors, etc.)
import cv2                       # OpenCV for video capture and image processing
from imutils import face_utils   # Contains helper functions to handle face landmarks
from pygame import mixer         # Used for playing alarm sounds

## Initialize the mixer and load the alarm sound file
mixer.init()
mixer.music.load("C:/Users/user/Downloads/music.wav")

## Start capturing video from the default webcam
cap = cv2.VideoCapture(0)

## Load Dlib’s face detector and pre-trained landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/user/Downloads/shape_predictor_68_face_landmarks.dat")

## Initialize counters and flags for different states
sleep = 0       # Number of consecutive frames for which eyes are closed (sleeping)
drowsy = 0      # Frames showing semi-closed eyes (drowsy state)
active = 0      # Frames showing open eyes (active/awake state)
status = ""     # Current status of the person (Active, Drowsy, or Sleeping)
color = (0, 0, 0)  # Color used for drawing text/status on the screen
alarm_on = False   # Boolean flag to track if the alarm sound is playing

## Function to compute the Euclidean distance between two facial landmark points
def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

## Function to detect blinking or eye status
# Takes coordinates of 6 eye landmarks:
# a and f are horizontal eye corners
# b, c, d, e are points for upper and lower eyelids
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)      # Sum of distances between upper and lower eyelids
    down = compute(a, f)                    # Distance between eye corners
    aspect_ratio = up / (2.0 * down)        # Eye aspect ratio (EAR)

    # The thresholds determine whether the eyes are open, drowsy, or closed
    if aspect_ratio > 0.25:
        return 2    # Eyes open (active)
    elif aspect_ratio > 0.21 and aspect_ratio <= 0.25:
        return 1    # Eyes partially closed (drowsy)
    else:
        return 0    # Eyes closed (sleeping)