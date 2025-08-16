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

# Try these ports in order - focus on COM4 since it shows "Access is denied" rather than "not found"
ARDUINO_PORTS = ['COM4']  # Reduced to just the port that seems to exist but might be busy
BAUD_RATE = 9600
SERVO_PIN = '9'  # Pin number for servo on Arduino
arduino_connection = None  # Global variable to hold the Arduino connection
RETRY_DELAY = 2  # Seconds to wait between connection attempts
MAX_RETRIES = 3  # Maximum number of retries before giving up on a reconnect attempt

def initialize_arduino():
    """Establishes a connection to the Arduino and returns the connection object"""
    global arduino_connection
    
    # Skip if Arduino is disabled
    if not USE_ARDUINO:
        print("Arduino usage is disabled. Running in display-only mode.")
        return None
    
    # Try each port in the list until one works
    for port in ARDUINO_PORTS:
        for attempt in range(MAX_RETRIES):
            try:
                print(f"Connecting to Arduino on {port} (attempt {attempt+1}/{MAX_RETRIES})...")
                
                # Close any existing connection first
                if arduino_connection is not None and arduino_connection.is_open:
                    try:
                        arduino_connection.close()
                        print("Closed previous connection")
                    except Exception as e:
                        print(f"Error closing previous connection: {e}")
                
                # Try to connect with a longer timeout for stability
                arduino_connection = serial.Serial(port=port, baudrate=BAUD_RATE, timeout=1.0)
                print(f"Connected to Arduino on {port}")
                
                # Give the Arduino time to reset and initialize
                time.sleep(2.0)  # Increased wait time
                
                # Clear any initial buffer and don't wait too long
                start_time = time.time()
                response_received = False
                
                # Try to read any initial message from Arduino
                while arduino_connection.in_waiting and time.time() - start_time < 2.0:
                    try:
                        line = arduino_connection.readline().decode('utf-8').strip()
                        if line:
                            print(f"Arduino says: {line}")
                            response_received = True
                    except Exception as e:
                        # If there's an error reading, just clear the buffer
                        print(f"Error reading from Arduino: {e}")
                        try:
                            arduino_connection.reset_input_buffer()
                        except:
                            pass
                        break
                
                # Return the connection even if we didn't get a response
                # The Arduino might not be sending anything initially
                return arduino_connection
                
            except serial.SerialException as e:
                if "PermissionError" in str(e) or "Access is denied" in str(e):
                    print(f"Port {port} is busy. Waiting {RETRY_DELAY} seconds before retry...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"Failed to connect to Arduino on {port}: {e}")
                    break  # Don't retry on other errors
            except Exception as e:
                print(f"Unexpected error connecting to Arduino on {port}: {e}")
                break  # Don't retry on unexpected errors
    
    print("Could not connect to Arduino on any available port. Try running with --no-arduino flag if needed.")
    return None

# --- Helpers ---

def send_signal(pin: str, value: float, is_servo: bool = False) -> None:
    """
    Sends a signal to the Arduino to control a pin.
    
    Value should be a decimal between 0 and 1.
    If is_servo=True, sends the value directly without a pin number (for servo_control.ino).
    """
    # Skip if Arduino is disabled
    if not USE_ARDUINO:
        return
        
    global arduino_connection
    
    # Only attempt to reconnect once per main loop iteration
    connection_attempts = 0
    
    # If no connection or connection is closed, try to initialize
    if arduino_connection is None or not arduino_connection.is_open:
        if connection_attempts < 1:  # Limit reconnection attempts
            print("No active Arduino connection. Attempting to reconnect...")
            arduino_connection = initialize_arduino()
            connection_attempts += 1
            
        if arduino_connection is None:
            print("Failed to send signal - no Arduino connection")
            return
    
    try:
        # For servo control, we just send the value directly
        if is_servo:
            command = f"{value}\n"
        else:
            command = f"{pin},{value}\n"
            
        arduino_connection.write(command.encode('utf-8'))
        print(f"Sent to Arduino: {command.strip()}")
        
        # Quick check for response (non-blocking)
        time.sleep(0.1)  # Slightly longer wait
        if arduino_connection.in_waiting:
            # Only try to read for a short time
            response = arduino_connection.read(arduino_connection.in_waiting).decode('utf-8').strip()
            if response:
                print(f"Arduino response: {response}")
                
    except Exception as e:
        print(f"Error sending signal to Arduino: {e}")
        # Try to reset the connection next time
        try:
            if arduino_connection and arduino_connection.is_open:
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
if USE_ARDUINO:
    try:
        print("Attempting to connect to Arduino...")
        arduino_result = initialize_arduino()
        
        if arduino_result:
            print("Arduino connection established successfully.")
            
            # Give the Arduino IDE a chance to release the port if it was previously being used for uploads
            print("Waiting for Arduino to be fully ready...")
            time.sleep(2.0)
            
            # Test servo control
            print("Testing servo activation...")
            send_signal("", 0.8, is_servo=True)  # Test servo movement
        else:
            print("Warning: Could not connect to Arduino.")
            print("If the Arduino is connected:")
            print("1. Check that it's plugged in and recognized by your system")
            print("2. Make sure the Arduino IDE isn't using the port")
            print("3. Try unplugging and reconnecting the Arduino")
            print("4. Or use --no-arduino flag to run without hardware")
    except Exception as e:
        print(f"Error during Arduino initialization: {e}")
        print("Continuing without Arduino connection...")
else:
    print("Arduino support disabled via command line. Running in display-only mode.")

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
    signal_hold_duration = 0.1  # Hold the signal for just 0.1 seconds (much shorter)
    last_signal_time = 0  # Track when we last sent a signal to avoid flooding
    
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
        
        # If signal has expired, reset the value to 0.0
        if signal_expired:
            current_signal_value = 0.0
        
        # Visualize the current signal value
        draw_circle(frame, current_signal_value)
        
        # Only attempt to send signals if we're not in display-only mode
        if USE_ARDUINO and current_signal_value > CHOP_THRESHOLD:
            # Limit how often we send signals to avoid flooding the Arduino
            # Only send a new signal if it's been more than 0.5 seconds since the last one
            if current_time - last_signal_time > 0.5:
                send_signal("", current_signal_value, is_servo=True)
                last_signal_time = current_time  # Update the last signal time
                
                # Reset the signal immediately after sending to prevent rapid repeats
                # but keep visual feedback in the UI
                # current_signal_value = 0.0  # Uncomment to reset signal immediately
        
        # Display status messages based on current state
        if current_signal_value > 0:
            cv2.putText(frame, f'Downward movement intensity: {current_signal_value:.2f}', (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
            
            if current_signal_value > CHOP_THRESHOLD:
                cv2.putText(frame, 'SERVO ACTIVATED!', (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 230), 2)
            
            # Show countdown for how long the signal will remain
            if not chop_detected:  # Only show countdown when not actively moving
                time_left = max(0.0, signal_hold_duration - (current_time - last_chop_time))
                cv2.putText(frame, f'Signal hold: {time_left:.1f}s', (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        else:
            cv2.putText(frame, 'No downward movement detected - Controls OFF', (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        
        cv2.putText(frame, f'Move your hand DOWNWARD to control servo (threshold: {CHOP_THRESHOLD:.2f}).', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        cv2.putText(frame, 'Downward movement above threshold activates servo. Press x to exit.', (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        cv2.imshow('Gesture Circle', frame.astype(np.uint8))
        if (cv2.waitKey(1) & 0xFF) == ord('x'):
            break

# Reset servo to initial position and clean up
if USE_ARDUINO and arduino_connection is not None and arduino_connection.is_open:
    # Reset servo to initial position (no need for specific value, just below threshold)
    send_signal("", 0.0, is_servo=True)
    arduino_connection.close()
    print("Arduino connection closed")

cap.release()
cv2.destroyAllWindows()
print('Gesture demo stopped.')
