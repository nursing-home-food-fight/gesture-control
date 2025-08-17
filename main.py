import cv2
import mediapipe as mp
from typing import Any, Final, Tuple
import numpy as np
import serial
import time
import sys
import mss  # ADDED: For screen capturing

mp_hands: Any = mp.solutions.hands
mp_drawing: Any = mp.solutions.drawing_utils
mp_styles: Any = mp.solutions.drawing_styles

# --- NEW: Screen Capture Config ---
# Define the region of the screen to capture.
# You will likely need to adjust these values for your screen setup.
# An easy way to find coordinates is to use a simple screenshot tool that shows pixel coordinates.
SCREEN_REGION: Final[dict[str, int]] = {'top': 100, 'left': 100, 'width': 1280, 'height': 720}

# --- Config ---
# Frame dimensions are now derived from the screen capture region for consistency
FRAME_WIDTH: Final[int] = SCREEN_REGION['width']
FRAME_HEIGHT: Final[int] = SCREEN_REGION['height']
CIRCLE_COLOR: Final[Tuple[int, int, int]] = (0, 0, 255)   # red

CHOP_THRESHOLD: Final[float] = 0.5  # Minimum vertical movement to be considered a chop

# --- Smoothing Config ---
SMOOTHING_FACTOR: Final[float] = 0.5  # 0 = no smoothing, 1 = maximum smoothing
HAND_TIMEOUT: Final[float] = 2.0  # Seconds to keep using last known hand position
USE_SMOOTHING: Final[bool] = True  # Enable or disable smoothing


# --- Hand Tracking with Smoothing ---
class HandTracker:
    def __init__(self, smoothing_factor: float = SMOOTHING_FACTOR):
        """
        Initialize the hand tracker with smoothing.
        
        Args:
            smoothing_factor: 0 = no smoothing, 1 = maximum smoothing (0.5 is good default)
        """
        self.smoothing_factor = max(0.0, min(1.0, smoothing_factor))  # Clamp between 0 and 1
        self.last_hand_lms = None  # Last detected hand landmarks
        self.smooth_hand_lms = None  # Smoothed hand landmarks
        self.last_detection_time = time.time()  # When the hand was last detected
        self.hand_visible = False  # Whether hand is currently visible
        self.prev_positions = []  # Store previous wrist Y positions
        
    def update(self, hand_landmarks):
        """Update with new hand landmarks."""
        self.hand_visible = True
        self.last_detection_time = time.time()
        
        # Store raw landmarks
        self.last_hand_lms = hand_landmarks
        
        # Apply smoothing if we have previous data
        if self.smooth_hand_lms is None:
            # First detection, no smoothing possible yet
            self.smooth_hand_lms = hand_landmarks
        else:
            # Apply smoothing to each landmark
            for i, landmark in enumerate(hand_landmarks.landmark):
                smooth_landmark = self.smooth_hand_lms.landmark[i]
                # Smooth x, y, z coordinates
                smooth_landmark.x = self._smooth_value(smooth_landmark.x, landmark.x)
                smooth_landmark.y = self._smooth_value(smooth_landmark.y, landmark.y)
                smooth_landmark.z = self._smooth_value(smooth_landmark.z, landmark.z)
        
        return self.smooth_hand_lms
    
    def _smooth_value(self, prev_value: float, new_value: float) -> float:
        """Apply smoothing between previous and new values."""
        if not USE_SMOOTHING:
            return new_value  # No smoothing
        return prev_value * self.smoothing_factor + new_value * (1 - self.smoothing_factor)
    
    def get_landmarks(self):
        """
        Get the current hand landmarks.
        
        Returns:
            Hand landmarks if hand is visible or within timeout period, None otherwise.
        """
        # If hand is currently visible, use the smoothed landmarks
        if self.hand_visible:
            return self.smooth_hand_lms
        
        # If within timeout window, use the last known position
        if time.time() - self.last_detection_time < HAND_TIMEOUT and self.smooth_hand_lms is not None:
            return self.smooth_hand_lms
        
        # No valid landmarks available
        return None
    
    def check_visibility(self):
        """
        Update hand visibility status based on timeout.
        Should be called every frame.
        """
        if time.time() - self.last_detection_time > HAND_TIMEOUT:
            self.hand_visible = False

# Check for "--no-arduino" command line argument
USE_ARDUINO = "--no-arduino" not in sys.argv

# Try these ports in order - focus on COM4 since it shows "Access is denied" rather than "not found"
ARDUINO_PORTS = ['/dev/cu.usbmodem101']  # Reduced to just the port that seems to exist but might be busy
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
                
                # Try to read any initial message from Arduino
                while arduino_connection.in_waiting and time.time() - start_time < 2.0:
                    try:
                        line = arduino_connection.readline().decode('utf-8').strip()
                        if line:
                            print(f"Arduino says: {line}")
                    except Exception as e:
                        # If there's an error reading, just clear the buffer
                        print(f"Error reading from Arduino: {e}")
                        try:
                            arduino_connection.reset_input_buffer()
                        except Exception as e2:
                            print(f"Failed to reset input buffer: {e2}")
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
# REMOVED: Webcam Initialization
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)


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
    # Initialize hand tracker with smoothing
    hand_tracker = HandTracker(SMOOTHING_FACTOR)
    
    # Variables for chop motion tracking
    prev_hand_positions = []
    last_chop_time = 0
    current_signal_value = 0.0
    signal_hold_duration = 0.1  # Hold the signal for just 0.1 seconds (much shorter)
    last_signal_time = 0  # Track when we last sent a signal to avoid flooding
    
    # ADDED: Initialize the screen capture object
    with mss.mss() as sct:
        while True:
            # MODIFIED: Capture from screen instead of camera
            # Grab the screen data from the defined region
            sct_img = sct.grab(SCREEN_REGION)

            # Convert the raw BGRA image to a NumPy array
            webcam_frame = np.array(sct_img)
            
            # Convert from BGRA to BGR for OpenCV compatibility (discards alpha channel)
            webcam_frame = cv2.cvtColor(webcam_frame, cv2.COLOR_BGRA2BGR)
            
            # MODIFIED: Flipping is usually for mirroring a webcam and is not needed for screen capture.
            # webcam_frame = cv2.flip(webcam_frame, 1)

            frame = webcam_frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            # Check if we should reset the signal after hold duration
            current_time = time.time()
            signal_expired = (current_time - last_chop_time) > signal_hold_duration
            
            chop_detected = False
            chop_speed = 0.0
            
            # Update hand tracker visibility status
            hand_tracker.check_visibility()
            
            # Process new hand detection
            if results.multi_hand_landmarks:
                # Update the tracker with the first hand detected
                hand_tracker.update(results.multi_hand_landmarks[0])
            
            # Get smoothed hand landmarks (could be current or last known position)
            smooth_hand_lms = hand_tracker.get_landmarks()
            
            if smooth_hand_lms:
                # Draw the smoothed hand landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    smooth_hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )
                
                # Process hand movement using the smoothed hand landmarks
                movement_detected, position_change = detect_vertical_position(smooth_hand_lms, prev_hand_positions)
                if movement_detected:
                    chop_detected = True
                    chop_speed = position_change
                    # Update signal value and reset timer when movement exceeds threshold
                    if position_change > CHOP_THRESHOLD:
                        current_signal_value = position_change
                        last_chop_time = current_time
                
                # Add status indicator for smoothing
                if not hand_tracker.hand_visible:
                    # If using last known position, display indicator
                    cv2.putText(frame, "Using last known hand position", (frame.shape[1] - 350, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
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
                if smooth_hand_lms:
                    status = "Hand detected - Controls ready" if hand_tracker.hand_visible else "Using last known position"
                    cv2.putText(frame, status, (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
                else:
                    cv2.putText(frame, 'No hand detected - Controls OFF', (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
            
            cv2.putText(frame, f'Move your hand DOWNWARD to control servo (threshold: {CHOP_THRESHOLD:.2f}).', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
            cv2.putText(frame, 'Downward movement above threshold activates servo. Press x to exit.', (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
            
            # Show smoothing information
            smooth_status = f"Smoothing: {'ON' if USE_SMOOTHING else 'OFF'} (factor: {SMOOTHING_FACTOR:.1f})"
            timeout_status = f"Hand timeout: {HAND_TIMEOUT:.1f}s"
            cv2.putText(frame, smooth_status, (20, frame.shape[0] - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, timeout_status, (20, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow('Gesture Circle', frame.astype(np.uint8))
            if (cv2.waitKey(1) & 0xFF) == ord('x'):
                break

# Reset servo to initial position and clean up
if USE_ARDUINO and arduino_connection is not None and arduino_connection.is_open:
    # Reset servo to initial position (no need for specific value, just below threshold)
    send_signal("", 0.0, is_servo=True)
    arduino_connection.close()
    print("Arduino connection closed")

# REMOVED: No longer using a camera capture object
# cap.release()
cv2.destroyAllWindows()
print('Gesture demo stopped.')