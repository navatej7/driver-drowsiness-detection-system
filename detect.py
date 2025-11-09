## Import required libraries
import dlib                     # Used for face detection and landmark prediction
import numpy as np               # Used for numerical operations (arrays, vectors, etc.)
import cv2                       # OpenCV for video capture and image processing
from imutils import face_utils   # Contains helper functions to handle face landmarks
from pygame import mixer         # Used for playing alarm sounds

# Initialize the mixer module for playing alert sounds
mixer.init()
mixer.music.load("C:/Users/user/Downloads/music.wav")

# Start video capture from the default webcam
cap = cv2.VideoCapture(0)

# Load dlib's face detector and pretrained facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/user/Downloads/shape_predictor_68_face_landmarks.dat")

# Initialize counters and state variables
sleep = 0      # Counts frames when user is detected as sleeping
drowsy = 0     # Counts frames when user is drowsy
active = 0     # Counts frames when user is active
status = ""    # Display string for user state
color = (0, 0, 0)  # Color used to draw rectangle and text
alarm_on = False   # Flag to track if alarm sound is currently playing

# Function to compute Euclidean distance between two points
def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

# Function to determine eye state based on aspect ratio
# a–f represent the six landmark points around one eye
def blinked(a, b, c, d, e, f):
    # Vertical distances between eyelid points
    up = compute(b, d) + compute(c, e)
    # Horizontal distance between corners of the eye
    down = compute(a, f)
    aspect_ratio = up / (2.0 * down)

    # Threshold decision for eye state
    if aspect_ratio > 0.25:
        return 2  # Eye open (Active)
    elif aspect_ratio > 0.21 and aspect_ratio <= 0.25:
        return 1  # Eye partially closed (Drowsy)
    else:
        return 0  # Eye closed (Sleeping)

# Main loop for continuous real-time video capture
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)  # Detect faces in grayscale frame

    # Create a copy of the frame to draw rectangles and annotations
    face_frame = frame.copy()

    for face in faces:
        # Get facial bounding box coordinates
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # Draw rectangle around detected face
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (255, 0, 100), 2)

        # Detect facial landmark points
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Compute blink ratio for both eyes
        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38],
                             landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44],
                              landmarks[47], landmarks[46], landmarks[45])

        # Determine alertness based on blink ratio results
        if left_blink == 0 or right_blink == 0:
            # Eyes closed → increase sleep counter
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (0, 0, 255)
                # Play alarm if not already playing
                if not alarm_on:
                    mixer.music.play(-1)
                    alarm_on = True

        elif left_blink == 1 or right_blink == 1:
            # Eyes partially closed → increase drowsy counter
            drowsy += 1
            sleep = 0
            active = 0
            if drowsy > 6:
                status = "Drowsy !"
                color = (255, 0, 0)
                # Stop alarm if it was playing
                if alarm_on:
                    mixer.music.stop()
                    alarm_on = False

        else:
            # Eyes open → user is active
            active += 1
            sleep = 0
            drowsy = 0
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)
                # Stop alarm in case it’s playing
                if alarm_on:
                    mixer.music.stop()
                    alarm_on = False

        # Display current status text on the frame
        cv2.putText(frame, status, (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    # Display the video feed
    cv2.imshow("Frame", frame)

    # Break loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
