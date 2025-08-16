import cv2
import mediapipe as mp
import serial
import time
import numpy as np
from typing import Any, Final, Optional

# --- Arduino & Serial Config ---
# ❗️ IMPORTANT: Update this to your Arduino's port
ARDUINO_PORT: Final[str] = '/dev/cu.usbmodem2101'
BAUD_RATE: Final[int] = 9600
PIN_TO_CONTROL: Final[int] = 8

# --- MediaPipe & OpenCV Config ---
mp_hands: Any = mp.solutions.hands
mp_drawing: Any = mp.solutions.drawing_utils
mp_styles: Any = mp.solutions.drawing_styles
FRAME_WIDTH: Final[int] = 1280
FRAME_HEIGHT: Final[int] = 720

# --- Helper Functions ---
def send_arduino_command(arduino: serial.Serial, pin: int, state: int) -> None:
    """Sends a command to the Arduino in the format 'pin,state\\n'."""
    command = f"{pin},{state}\n"
    arduino.write(command.encode('utf-8'))
    print(f"Sent command: {command.strip()}")

def is_fist(hand_lms: Any) -> bool:
    """A simple fist detection logic based on fingertip proximity to the palm."""
    # Indices: 0=wrist, 8=index_tip, 12=middle_tip, 16=ring_tip, 20=pinky_tip
    finger_tips = [8, 12, 16, 20]
    palm_base = hand_lms.landmark[0] # Wrist landmark
    
    # Check if the fingertips are below the knuckle landmarks (a better metric for a fist)
    is_closed = True
    for tip_idx in finger_tips:
        tip_pos = hand_lms.landmark[tip_idx].y
        mcp_pos = hand_lms.landmark[tip_idx - 2].y # Metacarpophalangeal joint (knuckle)
        if tip_pos > mcp_pos:
            is_closed = False
            break
    return is_closed

# --- Main Application ---
arduino: Optional[serial.Serial] = None
try:
    # Connect to Arduino
    arduino = serial.Serial(port=ARDUINO_PORT, baudrate=BAUD_RATE, timeout=.1)
    print(f"Connected to Arduino on {ARDUINO_PORT}")
    time.sleep(2)  # Allow time for the connection to establish

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    led_is_on = False  # State variable to track the LED status

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1, # Optimize for one hand
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.flip(frame, 1) # Flip horizontally for a mirror view
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            fist_detected_this_frame = False
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                    if is_fist(hand_lms):
                        fist_detected_this_frame = True

            # --- State Change Logic ---
            # If a fist is detected and the LED is currently off, turn it on.
            if fist_detected_this_frame and not led_is_on:
                send_arduino_command(arduino, PIN_TO_CONTROL, 1)
                led_is_on = True
            
            # If no fist is detected and the LED is currently on, turn it off.
            elif not fist_detected_this_frame and led_is_on:
                send_arduino_command(arduino, PIN_TO_CONTROL, 0)
                led_is_on = False

            # Display status on the screen
            status = "FIST CLOSED - LED ON" if led_is_on else "FIST OPEN - LED OFF"
            color = (0, 255, 0) if led_is_on else (0, 0, 255)
            cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow('Hand Gesture LED Control', frame)

            if cv2.waitKey(5) & 0xFF == ord('x'):
                break

finally:
    if arduino and arduino.is_open:
        if led_is_on: # Turn off LED on exit
            send_arduino_command(arduino, PIN_TO_CONTROL, 0)
        arduino.close()
        print("Arduino connection closed.")
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
    print("Application stopped.")