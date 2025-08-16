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

CHOP_THRESHOLD: Final[float] = 0.5  # Minimum vertical movement to be considered a chop

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
        # Always send the command, threshold is now handled in the main loop
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

def detect_vertical_position(hand_lms: Any, prev_hand_positions: list) -> Tuple[bool, float]:
    """
    Tracks the vertical position of the hand and calculates the change in position.
    Only detects DOWNWARD motion (increasing Y value in the image).
    
    Returns: (movement_detected, position_change) where position_change is a value between 0 and 1
    representing how much the hand has moved downward
    """
    # Get wrist landmark for tracking position
    wrist = hand_lms.landmark[0]
    
    # Default values
    movement_detected = False
    position_change = 0.0
    
    # We need at least one previous position to calculate change
    if prev_hand_positions:
        prev_wrist_y = prev_hand_positions[-1]
        
        # Calculate vertical movement (positive means moving down in image coordinates)
        vertical_change = wrist.y - prev_wrist_y
        
        # Only consider downward movement (increasing Y)
        if vertical_change > 0:
            # Normalize the change to a 0-1 range
            # 0.005 would be a small movement, 0.05+ would be a large movement
            position_change = min(1.0, vertical_change / 0.05)
            
            # Consider any significant downward movement
            if position_change > 0.05:
                movement_detected = True
    
    # Update position history
    prev_hand_positions.append(wrist.y)
    # Keep only the last 5 positions
    if len(prev_hand_positions) > 5:
        prev_hand_positions.pop(0)
    
    return movement_detected, position_change

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
    # Variables for chop motion tracking
    prev_hand_positions = []
    last_chop_time = 0
    current_signal_value = 0.0
    signal_hold_duration = 1.0  # Hold the signal for 1 second
    
    while True:
        ret, webcam_frame = cap.read()
        if not ret:
            break
        webcam_frame = cv2.resize(webcam_frame, (FRAME_WIDTH, FRAME_HEIGHT))
        webcam_frame = cv2.flip(webcam_frame, 1)
        frame = webcam_frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        # Check if we should reset the signal after hold duration
        current_time = time.time()
        signal_expired = (current_time - last_chop_time) > signal_hold_duration
        
        chop_detected = False
        chop_speed = 0.0
        
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )
                movement_detected, position_change = detect_vertical_position(hand_lms, prev_hand_positions)
                if movement_detected:
                    chop_detected = True
                    chop_speed = max(chop_speed, position_change)  # Take the highest change if multiple hands
                    # Update signal value and reset timer when movement exceeds threshold
                    if position_change > CHOP_THRESHOLD:
                        current_signal_value = position_change
                        last_chop_time = current_time
        
        # Throttle rate of sending signals to avoid flooding Arduino
        should_send_signal = int(current_time * 5) % 2 == 0  # Send updates at reduced rate (every ~0.4 seconds)
        
        # If signal has expired, reset the value to 0.0
        # This will turn off the LED after the hold duration
        if signal_expired:
            current_signal_value = 0.0
        
        # Visualize the current signal value
        draw_circle(frame, current_signal_value)
        
        if should_send_signal:
            send_signal(CONTROL_PIN, current_signal_value)
        
        # Display status messages based on current state
        if current_signal_value > 0:
            cv2.putText(frame, f'Downward movement intensity: {current_signal_value:.2f}', (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
            
            # Show countdown for how long the signal will remain
            if not chop_detected:  # Only show countdown when not actively moving
                time_left = max(0.0, signal_hold_duration - (current_time - last_chop_time))
                cv2.putText(frame, f'Signal hold: {time_left:.1f}s', (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        else:
            cv2.putText(frame, 'No downward movement detected - LED OFF', (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        
        cv2.putText(frame, f'Move your hand DOWNWARD to control LED (threshold: {CHOP_THRESHOLD:.2f}).', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        cv2.putText(frame, 'Downward movement above threshold = LED on for 1 second. Press x to exit.', (20, 160),
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
