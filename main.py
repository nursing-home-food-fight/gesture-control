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

# Angle constants
MIN_ANGLE: Final[float] = 0.0  # Minimum angle value for normalization (straight arm)
MAX_ANGLE: Final[float] = 90.0  # Maximum angle value for normalization (90 degree bend)

CHOP_THRESHOLD: Final[float] = 0.5  # Keeping this for backward compatibility

# --- Smoothing Config ---
SMOOTHING_FACTOR: Final[float] = 0.5  # 0 = no smoothing, 1 = maximum smoothing
HAND_TIMEOUT: Final[float] = 2.0  # Seconds to keep using last known hand position
USE_SMOOTHING: Final[bool] = True  # Enable or disable smoothing

# --- Sampling Config ---
SAMPLE_RATE_HZ: Final[float] = 60.0  # How many samples per second to send to the Arduino
SAMPLE_INTERVAL: Final[float] = 1.0 / SAMPLE_RATE_HZ  # Time between samples in seconds
USE_INTERPOLATION: Final[bool] = True  # Whether to use linear interpolation between samples


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
ARDUINO_PORTS = ['COM4']  # Reduced to just the port that seems to exist but might be busy
BAUD_RATE = 9600
SERVO_PIN = '9'  # Pin number for servo on Arduino
arduino_connection = None  # Global variable to hold the Arduino connection
connection_attempts = 0  # Global counter for connection attempts
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
                while time.time() - start_time < 2.0:
                    try:
                        if arduino_connection and arduino_connection.in_waiting > 0:
                            line = arduino_connection.readline().decode('utf-8').strip()
                            if line:
                                print(f"Arduino says: {line}")
                        else:
                            break  # No more data or no connection
                    except Exception as e:
                        # If there's an error reading, just clear the buffer
                        print(f"Error reading from Arduino: {e}")
                        try:
                            if arduino_connection:
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

def remap(value: float) -> float:
    """
    Remaps a value from the observed hand angle range (0.1-0.4) to a full servo range (0.0-1.0).
    
    This ensures the servo uses most of its motion range (1-89 degrees after Arduino mapping).
    
    Returns:
        The remapped value as a float between 0.01 and 0.99.
    """
    # Define the input range we typically observe
    in_min = 0.1
    in_max = 0.4
    
    # Define the output range we want (slightly smaller than 0-1 to avoid servo limits)
    out_min = 0.01  # Maps to 1 degree
    out_max = 0.99  # Maps to 89 degrees
    
    # Handle edge cases
    if value <= in_min:
        return out_min
    if value >= in_max:
        return out_max
    
    # Linear interpolation
    return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)

def send_signal(pin: str, value: float, is_servo: bool = False) -> None:
    """
    Sends a signal to the Arduino to control a pin.
    
    Value should be a decimal between 0 and 1.
    If is_servo=True, sends the value directly without a pin number (for servo_control.ino).
    """
    # Skip if Arduino is disabled
    if not USE_ARDUINO:
        return

    value = remap(value)

    global arduino_connection
    
    # Use a simple global counter instead of function attributes
    global connection_attempts
    
    # If no connection or connection is closed, try to initialize
    if arduino_connection is None or not arduino_connection.is_open:
        if connection_attempts < 1:  # Limit reconnection attempts
            print("No active Arduino connection. Attempting to reconnect...")
            arduino_connection = initialize_arduino()
            connection_attempts += 1
            
        if arduino_connection is None:
            print("Failed to send signal - no Arduino connection")
            return
    else:
        # Reset counter when connection is good
        connection_attempts = 0
    
    try:
        # For servo control, we just send the value directly
        if is_servo:
            command = f"{value:.2f}\n"  # Limit precision to reduce packet size
        else:
            command = f"{pin},{value:.2f}\n"
            
        arduino_connection.write(command.encode('utf-8'))
        
        # Read any available data to keep the buffer clear (non-blocking)
        if arduino_connection.in_waiting > 0:
            try:
                response = arduino_connection.read(arduino_connection.in_waiting).decode('utf-8')
                if response.strip():  # Only print if there's actual content
                    print(f"Arduino response: {response.strip()}")
            except Exception as read_error:
                print(f"Error reading from Arduino: {read_error}")
                
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

def calculate_hand_angle(hand_lms: Any) -> float:
    """
    Calculate the angle between the wrist, elbow and shoulder.
    
    In MediaPipe hands, we don't have shoulder and elbow, so we'll use:
    - Wrist (landmark 0)
    - Middle of palm (approximated as the average of landmarks 5, 9, 13, 17)
    - Base of pinky (landmark 17) as a third point to calculate angle

    Returns: A value between 0 and 1 representing the normalized angle
    """
    # Get wrist position
    wrist = hand_lms.landmark[0]
    
    # Approximate middle of palm as the average of base landmarks of each finger
    palm_x = sum(hand_lms.landmark[i].x for i in [5, 9, 13, 17]) / 4
    palm_y = sum(hand_lms.landmark[i].y for i in [5, 9, 13, 17]) / 4
    palm_z = sum(hand_lms.landmark[i].z for i in [5, 9, 13, 17]) / 4
    
    # Base of pinky as third point
    pinky_base = hand_lms.landmark[17]
    
    # Calculate vectors
    vector1 = [palm_x - wrist.x, palm_y - wrist.y, palm_z - wrist.z]
    vector2 = [pinky_base.x - wrist.x, pinky_base.y - wrist.y, pinky_base.z - wrist.z]
    
    # Calculate norms using manual calculation to avoid type issues
    norm1 = np.sqrt(sum(v*v for v in vector1))
    norm2 = np.sqrt(sum(v*v for v in vector2))
    
    # Normalize vectors (safely)
    if norm1 > 0:
        vector1 = [v/norm1 for v in vector1]
    if norm2 > 0:
        vector2 = [v/norm2 for v in vector2]
    
    # Calculate dot product manually
    dot_product = sum(v1*v2 for v1, v2 in zip(vector1, vector2))
    
    # Clamp the dot product to [-1, 1] to avoid numerical issues
    dot_product = max(-1.0, min(1.0, dot_product))
    
    # Calculate angle in degrees
    angle = np.arccos(dot_product) * 180 / np.pi
    
    # Normalize angle to 0-1 range using min and max angles
    normalized_angle = max(0.0, min(1.0, (angle - MIN_ANGLE) / (MAX_ANGLE - MIN_ANGLE)))
    
    return normalized_angle

def detect_hand_angle(hand_lms: Any) -> float:
    """
    Calculate the angle of the hand relative to the wrist and return a normalized value.
    
    Returns: A value between 0 and 1 representing the normalized hand angle
    """
    # Calculate the normalized angle
    angle_value = calculate_hand_angle(hand_lms)
    
    return angle_value

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
    
    # Variables for tracking
    current_signal_value = 0.0
    last_sent_value = 0.0
    last_sample_time = time.time()
    next_sample_time = last_sample_time + SAMPLE_INTERVAL
    interpolation_start_value = 0.0
    interpolation_target_value = 0.0
    
    # ADDED: Initialize the screen capture object
    with mss.mss() as sct:
        # Add a safety counter and last_error_time to prevent error spam
        error_count = 0
        last_error_time = 0
        max_errors_per_second = 5
        
        while True:
            try:
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
                
                # Get current time
                current_time = time.time()
            except Exception as e:
                # Rate limit errors to prevent console spam
                if time.time() - last_error_time > 1.0:  # Reset counter every second
                    error_count = 0
                    last_error_time = time.time()
                    
                error_count += 1
                if error_count <= max_errors_per_second:
                    print(f"Error in screen capture processing: {e}")
                elif error_count == max_errors_per_second + 1:
                    print("Too many errors, suppressing further messages for this second")
                    
                # Try to continue by sleeping a short time
                time.sleep(0.1)
                continue
            
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
                
                # Calculate the hand angle using the smoothed landmarks
                hand_angle = detect_hand_angle(smooth_hand_lms)
                
                # Update the detected angle value
                interpolation_target_value = hand_angle
                
                # Calculate where we are in the interpolation between samples
                time_since_last_sample = current_time - last_sample_time
                
                # Prevent division by zero or negative values
                if SAMPLE_INTERVAL > 0:
                    progress = min(1.0, max(0.0, time_since_last_sample / SAMPLE_INTERVAL))
                else:
                    progress = 1.0
                
                # Apply linear interpolation between the last sent value and the target
                if USE_INTERPOLATION:
                    # Ensure all values are valid floats
                    try:
                        current_signal_value = interpolation_start_value + (interpolation_target_value - interpolation_start_value) * progress
                    except Exception as e:
                        print(f"Interpolation error: {e}, using target value directly")
                        current_signal_value = interpolation_target_value
                else:
                    current_signal_value = interpolation_target_value
                
                # Print current interpolated value less frequently to avoid console spam
                # Using integer division to avoid floating-point modulo issues
                if int(current_time * 10) % 5 == 0:  # Approximately every 0.5 seconds
                    print(f"Hand angle value (signal): {current_signal_value:.2f}, Target: {interpolation_target_value:.2f}")
                
                # Add status indicator for smoothing
                if not hand_tracker.hand_visible:
                    # If using last known position, display indicator
                    cv2.putText(frame, "Using last known hand position", (frame.shape[1] - 350, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Reset signal value to 0 if no hand is detected
            if not smooth_hand_lms:
                current_signal_value = 0.0
            
            # Visualize the current signal value
            draw_circle(frame, current_signal_value)
            
            # Check if it's time for a new sample
            if current_time >= next_sample_time:
                # Save current value as the start point for the next interpolation
                interpolation_start_value = current_signal_value
                
                # At sample time, the target becomes the current value
                if smooth_hand_lms:
                    # Use the target value as our new setpoint
                    sample_value = interpolation_target_value
                else:
                    # No hand detected, use 0.0
                    sample_value = 0.0
                
                # Only attempt to send signals if we're not in display-only mode
                if USE_ARDUINO:
                    try:
                        send_signal("", sample_value, is_servo=True)
                        print(f"SAMPLED: Sending value: {sample_value:.2f} at {SAMPLE_RATE_HZ}Hz")
                    except Exception as serial_error:
                        print(f"Serial communication error: {serial_error}")
                        # Try to recover the Arduino connection
                        try:
                            if arduino_connection and arduino_connection.is_open:
                                arduino_connection.close()
                        except Exception as close_error:
                            print(f"Error closing Arduino connection: {close_error}")
                        arduino_connection = None
                
                # Update timing for next sample
                last_sample_time = current_time
                next_sample_time = current_time + SAMPLE_INTERVAL
                last_sent_value = sample_value
                    
            # Display status messages based on current state
            if smooth_hand_lms:
                # Show the current hand angle value (signal)
                cv2.putText(frame, f'Hand angle value: {current_signal_value:.2f}', (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
                
                status = "Hand detected - Controls active" if hand_tracker.hand_visible else "Using last known position"
                cv2.putText(frame, status, (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
                        
                # Show servo activation message
                cv2.putText(frame, 'Signal sent to servo (or printed in console)', (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 230), 2)
            else:
                cv2.putText(frame, 'No hand detected - Controls OFF', (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
            
            cv2.putText(frame, 'Change hand angle to control servo (0 = straight, 1 = bent 90°).', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
            cv2.putText(frame, f'Signal sent at {SAMPLE_RATE_HZ:.1f}Hz with interpolation. Press x to exit.', (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
            
            # Show hand tracking information
            smooth_status = f"Smoothing: {'ON' if USE_SMOOTHING else 'OFF'} (factor: {SMOOTHING_FACTOR:.1f})"
            timeout_status = f"Hand timeout: {HAND_TIMEOUT:.1f}s"
            
            # Show sampling information
            sample_status = f"Sample rate: {SAMPLE_RATE_HZ:.1f}Hz (every {SAMPLE_INTERVAL*1000:.1f}ms)"
            interp_status = f"Interpolation: {'ON' if USE_INTERPOLATION else 'OFF'}, Next sample in: {(next_sample_time - current_time)*1000:.0f}ms"
            
            # Add status about the signal being sent
            if smooth_hand_lms:
                signal_status = f"Signal: Current={current_signal_value:.2f}, Target={interpolation_target_value:.2f}, Last Sent={last_sent_value:.2f}"
            else:
                signal_status = "No hand detected - no signals sent"
            
            cv2.putText(frame, smooth_status, (20, frame.shape[0] - 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, sample_status, (20, frame.shape[0] - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, signal_status, (20, frame.shape[0] - 30),
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