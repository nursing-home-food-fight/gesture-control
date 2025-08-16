import cv2
import mediapipe as mp
from typing import Any, Final, Tuple
import numpy as np
import serial
import time
import sys

mp_hands: Any = mp.solutions.hands
mp_drawing: Any = mp.solutions.drawing_utils
mp_styles: Any = mp.solutions.drawing_styles

# --- Config ---
FRAME_WIDTH: Final[int] = 1280
FRAME_HEIGHT: Final[int] = 720
CIRCLE_COLOR: Final[Tuple[int, int, int]] = (0, 0, 255)   # red

# Check for "--no-arduino" command line argument
USE_ARDUINO = "--no-arduino" not in sys.argv

ARDUINO_PORT = 'COM3'
BAUD_RATE = 9600
CONTROL_PIN = '6'
arduino_connection = None  # Global variable to hold the Arduino connection
led_is_on = False  # Global variable to track if LED is currently on

def initialize_arduino():
    """Establishes a connection to the Arduino and returns the connection object"""
    global arduino_connection
    
    # Skip if Arduino is disabled
    if not USE_ARDUINO:
        print("Arduino usage is disabled. Running in display-only mode.")
        return None
        
    try:
        print(f"Connecting to Arduino on {ARDUINO_PORT}...")
        arduino_connection = serial.Serial(port=ARDUINO_PORT, baudrate=BAUD_RATE, timeout=0.5)
        print(f"Connected to Arduino on {ARDUINO_PORT}")
        time.sleep(1)  # Give the Arduino time to reset, but not too long
        
        # Clear any initial buffer and don't wait too long
        start_time = time.time()
        while arduino_connection.in_waiting and time.time() - start_time < 1.0:
            try:
                line = arduino_connection.readline().decode('utf-8').strip()
                print(f"Arduino says: {line}")
            except Exception as e:
                # If there's an error reading, just clear the buffer
                print(f"Error reading from Arduino: {e}")
                arduino_connection.reset_input_buffer()
                break
        
        return arduino_connection
    except Exception as e:
        print(f"Failed to connect to Arduino: {e}")
        return None

# --- Helpers ---

def send_signal(pin: str, value: float) -> None:
    """
    Sends a signal to the Arduino to control a pin.
    
    Value should be a decimal between 0 and 1
    """
    # Skip if Arduino is disabled
    if not USE_ARDUINO:
        return
        
    global arduino_connection
    if arduino_connection is None or not arduino_connection.is_open:
        print("No active Arduino connection. Attempting to reconnect...")
        arduino_connection = initialize_arduino()
        if arduino_connection is None:
            print("Failed to send signal - no Arduino connection")
            return
    
    try:
        command = f"{pin},{value}\n"
        arduino_connection.write(command.encode('utf-8'))
        print(f"Sent to Arduino: {command.strip()}")
        
        # Quick check for response (non-blocking)
        time.sleep(0.05)  # Very brief wait
        if arduino_connection.in_waiting:
            # Only try to read for a short time
            response = arduino_connection.read(arduino_connection.in_waiting).decode('utf-8').strip()
            if response:
                print(f"Arduino response: {response}")
    except Exception as e:
        print(f"Error sending signal to Arduino: {e}")
        # Try to reset the connection next time
        try:
            arduino_connection.close()
        except Exception as e2:
            print(f"Error closing Arduino connection: {e2}")
        arduino_connection = None

def draw_circle(img: Any, intensity: float = 1.0) -> None:
    h, w = img.shape[:2]
    
    # Adjust radius based on intensity
    min_radius = int(min(w, h) * 0.10)  # Minimum radius
    max_radius = int(min(w, h) * 0.25)  # Maximum radius
    radius = min_radius + int((max_radius - min_radius) * intensity)
    
    # Adjust thickness and color based on intensity
    thickness = int(18 * intensity) + 2  # Minimum thickness of 2
    
    # Adjust color intensity - make it brighter with higher intensity
    # The color will go from dim red to bright red
    intensity_color = (0, 0, int(255 * intensity))
    
    cv2.circle(img, (w // 2, h // 2), radius, intensity_color, thickness=thickness)

def get_fist_closedness(hand_lms: Any) -> Tuple[bool, float]:
    # Detect fist and measure how closed it is
    # Indices: 0=wrist, 4=thumb_tip, 8=index_tip, 12=middle_tip, 16=ring_tip, 20=pinky_tip
    tips = [4, 8, 12, 16, 20]
    palm = hand_lms.landmark[0]
    closed = 0
    total_dist = 0
    max_dist = 0.25  # Maximum expected distance
    
    for tip_idx in tips:
        tip = hand_lms.landmark[tip_idx]
        dist = ((tip.x - palm.x) ** 2 + (tip.y - palm.y) ** 2) ** 0.5
        total_dist += dist
        if dist < 0.18:
            closed += 1
    
    # Calculate closedness as a value between 0 and 1
    # Lower total_dist means more closed fist
    closedness = max(0.0, min(1.0, 1.0 - (total_dist / (len(tips) * max_dist))))
    
    return closed >= 3, closedness

cv2.destroyAllWindows()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)



# Try to initialize Arduino, but don't let it hang the program
try:
    print("Attempting to connect to Arduino...")
    arduino_result = initialize_arduino()
    if arduino_result:
        # Send quick test signals
        print("Sending test signal to Arduino...")
        send_signal(CONTROL_PIN, 1.0)  # Test with full brightness
        time.sleep(0.5)
        send_signal(CONTROL_PIN, 0.0)  # Turn off
    else:
        print("Warning: Could not connect to Arduino. Will try again when needed.")
except Exception as e:
    print(f"Error during Arduino initialization: {e}")
    print("Continuing without Arduino connection...")

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
) as hands:
    while True:
        ret, webcam_frame = cap.read()
        if not ret:
            break
        webcam_frame = cv2.resize(webcam_frame, (FRAME_WIDTH, FRAME_HEIGHT))
        webcam_frame = cv2.flip(webcam_frame, 1)
        frame = webcam_frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        fist_detected = False
        closedness = 0.0
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )
                is_fist, fist_closedness = get_fist_closedness(hand_lms)
                if is_fist:
                    fist_detected = True
                    closedness = max(closedness, fist_closedness)  # Take the highest closedness if multiple hands
        
        # Throttle rate of sending signals to avoid flooding Arduino
        should_send_signal = int(time.time() * 5) % 2 == 0  # Send updates at reduced rate (every ~0.4 seconds)
        
        # Visualize and send the closedness value if fist detected
        if fist_detected:
            draw_circle(frame, closedness)
            
            if should_send_signal:
                send_signal(CONTROL_PIN, closedness)
            
            # Display the closedness value on screen
            cv2.putText(frame, f'Fist closedness: {closedness:.2f}', (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        
        # If no fist is detected, turn off the LED
        elif should_send_signal:
            send_signal(CONTROL_PIN, 0.0)
            
            # Add a status message
            cv2.putText(frame, 'No fist detected - LED OFF', (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
            
        cv2.putText(frame, 'Make a fist to control intensity. Tighter fist = higher value.', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        cv2.putText(frame, 'Circle size, color, and thickness all reflect intensity. Press x to exit.', (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        cv2.imshow('Gesture Circle', frame.astype(np.uint8))
        if (cv2.waitKey(1) & 0xFF) == ord('x'):
            break

# Turn off the LED and clean up
if USE_ARDUINO and arduino_connection is not None and arduino_connection.is_open:
    send_signal(CONTROL_PIN, 0.0)  # Turn off the LED
    arduino_connection.close()
    print("Arduino connection closed")

cap.release()
cv2.destroyAllWindows()
print('Gesture demo stopped.')
