from typing import Any
from typing import Dict, List, Tuple
from mediapipe.tasks import python as mp_python
import mediapipe as mp
import numpy as np
import cv2
import logging
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Constants for circle targets
LEFT_CIRCLE_POS = (0.25, 0.5)  # (x, y) as percentage of screen width and height
RIGHT_CIRCLE_POS = (0.75, 0.5)  # (x, y) as percentage of screen width and height
CIRCLE_RADIUS_PERCENT = 0.1  # Radius as percentage of screen width
CIRCLE_COLOR = (0, 0, 255)  # Red color (BGR)
HIGHLIGHT_COLOR = (0, 255, 255)  # Yellow color (BGR)
TRIGGERED_COLOR = (0, 255, 0)  # Green color (BGR)
CIRCLE_THICKNESS = 3  # Default thickness
# Pointing detection parameters
POINTING_DOT_THRESHOLD = 0.9  # Higher value requires more precise pointing (was 0.8)

# Create MediaPipe hand landmarker
base_options = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
options = mp_python.vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = mp_python.vision.HandLandmarker.create_from_options(options)
logger.info("MediaPipe hand landmarker created successfully")

class Point2D(BaseModel):
    x: float
    y: float
    
class HandLandmarkResult(BaseModel):
    landmarks: List[Point2D]
    handedness: str

def draw_landmarks_manually(frame: np.ndarray, landmarks: List[Point2D]) -> None:
    """Draw hand landmarks manually on the frame."""
    # Define hand connections (index pairs)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # ring finger
        (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
        (5, 9), (9, 13), (13, 17)  # palm
    ]
    
    h, w, _ = frame.shape
    
    # Draw landmarks with bigger, more visible circles
    for landmark in landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)  # Increased size
        cv2.circle(frame, (x, y), 9, (255, 255, 255), 1)  # White outline
    
    # Draw connections with thicker lines
    for start_idx, end_idx in connections:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
            end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
            cv2.line(frame, start_point, end_point, (0, 0, 255), 3)  # Thicker lines

def detect_hands(frame: np.ndarray) -> List[HandLandmarkResult]:
    """Detects hands in the given frame and returns structured hand landmark results.
    
    Args:
        frame: The input video frame
        
    Returns:
        List of HandLandmarkResult objects containing landmarks and handedness
    """
    # Convert BGR to RGB (MediaPipe expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Run hand detection
    results = detector.detect(image)
    
    # Process the results into our Pydantic model
    hand_results = []
    
    if results.hand_landmarks:
        for i, hand_landmarks in enumerate(results.hand_landmarks):
            # Extract landmarks
            landmarks = [
                Point2D(x=landmark.x, y=landmark.y)
                for landmark in hand_landmarks
            ]
            
            # Extract handedness if available
            handedness = "Unknown"
            if results.handedness and i < len(results.handedness):
                handedness = results.handedness[i][0].category_name
            
            # Create a HandLandmarkResult object
            hand_result = HandLandmarkResult(
                landmarks=landmarks,
                handedness=handedness
            )
            
            hand_results.append(hand_result)
            
    return hand_results

def draw_hands(frame: np.ndarray, hand_results: List[HandLandmarkResult], overlay_data: Dict[str, Any]) -> None:
    """Draws hand landmarks and adds text overlays.
    
    Args:
        frame: The frame to draw on
        hand_results: List of HandLandmarkResult objects
        overlay_data: Dictionary for text overlays
    """
    num_hands = len(hand_results)
    
    if num_hands > 0:
        # Add text overlay to indicate hands detected
        overlay_data["texts"].append((
            f"Hands detected: {num_hands}", 
            (10, 60),  # position (x, y)
            0.7,       # font scale
            (0, 255, 0)  # color (green)
        ))
        
        # Draw hand landmarks on the frame
        for i, hand_result in enumerate(hand_results):
            logger.debug(f"Drawing hand {i+1} with {len(hand_result.landmarks)} landmarks")
            draw_landmarks_manually(frame, hand_result.landmarks)
            
            # Display handedness
            overlay_data["texts"].append((
                f"{hand_result.handedness} Hand", 
                (10, 90 + i * 30),
                0.6,
                (255, 0, 255)
            ))
    else:
        logger.debug("No hands detected")
        overlay_data["texts"].append((
            "No hands detected", 
            (10, 60),
            0.7,
            (0, 0, 255)
        ))

def is_pointing(landmarks: List[Point2D]) -> bool:
    """Detects if the hand is in a pointing gesture.
    
    A pointing gesture is defined as having the index finger extended while
    other fingers are curled in.
    
    Args:
        landmarks: List of hand landmarks
        
    Returns:
        True if pointing gesture detected, False otherwise
    """
    # We need at least 21 landmarks for a complete hand
    if len(landmarks) < 21:
        return False
        
    # Get key landmarks for the hand
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    index_mcp = landmarks[5]  # Index finger base
    index_tip = landmarks[8]  # Index finger tip
    
    # Check if index finger is extended
    index_extended = index_tip.y < index_mcp.y
    
    # Check if other fingers are curled in (y position of tip higher than MCP)
    middle_curled = landmarks[12].y > landmarks[9].y
    ring_curled = landmarks[16].y > landmarks[13].y
    pinky_curled = landmarks[20].y > landmarks[17].y
    
    # For thumb, we check if it's closer to the palm than extended
    thumb_curled = calculate_distance(thumb_tip, wrist) < calculate_distance(index_tip, wrist) * 0.8
    
    # Return True if index is extended and at least 2 other fingers are curled
    curled_count = sum([middle_curled, ring_curled, pinky_curled, thumb_curled])
    return index_extended and curled_count >= 2
    
def is_trigger_pulled(landmarks: List[Point2D]) -> bool:
    """Detects if the user is pulling a trigger while pointing.
    
    A trigger pull is defined as having the middle finger curled significantly
    more than in the normal pointing gesture.
    
    Args:
        landmarks: List of hand landmarks
        
    Returns:
        True if trigger pull detected, False otherwise
    """
    # We need at least 21 landmarks for a complete hand
    if len(landmarks) < 21:
        return False
    
    # First check if the hand is in a pointing gesture
    if not is_pointing(landmarks):
        return False
    
    # Check if middle finger is deeply curled (closer to palm)
    middle_mcp = landmarks[9]  # Middle finger base
    middle_pip = landmarks[10]  # Middle finger middle joint
    middle_tip = landmarks[12]  # Middle finger tip
    
    # Calculate how curled the middle finger is
    middle_curl_ratio = calculate_distance(middle_tip, middle_mcp) / calculate_distance(middle_pip, middle_mcp)
    
    # When the middle finger is deeply curled, this ratio will be smaller
    return middle_curl_ratio < 1.2  # Threshold for detecting trigger pull

def calculate_distance(p1: Point2D, p2: Point2D) -> float:
    """Calculate the Euclidean distance between two points.
    
    Args:
        p1: First point
        p2: Second point
        
    Returns:
        Distance between the two points
    """
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

def get_pointing_direction(landmarks: List[Point2D]) -> np.ndarray:
    """Get the direction vector of pointing.
    
    Args:
        landmarks: List of hand landmarks
        
    Returns:
        Unit vector in the pointing direction
    """
    # Use the index finger base to tip as the pointing direction
    index_mcp = landmarks[5]  # Index finger base
    index_tip = landmarks[8]  # Index finger tip
    
    # Calculate direction vector
    direction = np.array([index_tip.x - index_mcp.x, index_tip.y - index_mcp.y])
    
    # Normalize to get unit vector
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    
    return direction

def draw_target_circles(frame: np.ndarray, pointing_hands: List[Tuple[List[Point2D], np.ndarray]]) -> None:
    """Draw the target circles and highlight them if pointed at.
    
    Args:
        frame: The frame to draw on
        pointing_hands: List of tuples containing (landmarks, direction) for each pointing hand
    """
    h, w = frame.shape[:2]
    
    # Calculate circle positions and radius in pixels
    left_center = (int(LEFT_CIRCLE_POS[0] * w), int(LEFT_CIRCLE_POS[1] * h))
    right_center = (int(RIGHT_CIRCLE_POS[0] * w), int(RIGHT_CIRCLE_POS[1] * h))
    radius = int(CIRCLE_RADIUS_PERCENT * min(w, h))
    
    # Add labels to circles
    cv2.putText(frame, "Target 1", 
                (left_center[0] - radius//2, left_center[1] - radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CIRCLE_COLOR, 2)
    cv2.putText(frame, "Target 2", 
                (right_center[0] - radius//2, right_center[1] - radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CIRCLE_COLOR, 2)
    
    # Track circle states: 0=normal, 1=highlighted, 2=triggered
    left_state = 0
    right_state = 0
    
    # Check if any hand is pointing at the circles
    for landmarks, direction in pointing_hands:
        # Get index finger tip position as the origin of pointing
        index_tip = landmarks[8]
        origin = np.array([index_tip.x * w, index_tip.y * h])
        
        # Check if trigger is being pulled
        trigger_pulled = is_trigger_pulled(landmarks)
        
        # Check if pointing at left circle
        left_vector = np.array([left_center[0] - origin[0], left_center[1] - origin[1]])
        left_dist = np.linalg.norm(left_vector)
        if left_dist > 0:
            left_vector = left_vector / left_dist
            left_dot = np.dot(direction, left_vector)
            
            # If pointing in the direction of the circle and close enough
            # Using more strict threshold from constant
            if left_dot > POINTING_DOT_THRESHOLD and left_dist < w * 0.5:  # Reduced distance threshold
                if trigger_pulled:
                    left_state = 2  # Triggered
                else:
                    left_state = max(left_state, 1)  # Highlighted
                
                # Draw a line from finger to circle
                finger_pos = (int(index_tip.x * w), int(index_tip.y * h))
                line_color = TRIGGERED_COLOR if trigger_pulled else HIGHLIGHT_COLOR
                cv2.line(frame, finger_pos, left_center, line_color, 2)
        
        # Check if pointing at right circle
        right_vector = np.array([right_center[0] - origin[0], right_center[1] - origin[1]])
        right_dist = np.linalg.norm(right_vector)
        if right_dist > 0:
            right_vector = right_vector / right_dist
            right_dot = np.dot(direction, right_vector)
            
            # If pointing in the direction of the circle and close enough
            if right_dot > POINTING_DOT_THRESHOLD and right_dist < w * 0.5:
                if trigger_pulled:
                    right_state = 2  # Triggered
                else:
                    right_state = max(right_state, 1)  # Highlighted
                
                # Draw a line from finger to circle
                finger_pos = (int(index_tip.x * w), int(index_tip.y * h))
                line_color = TRIGGERED_COLOR if trigger_pulled else HIGHLIGHT_COLOR
                cv2.line(frame, finger_pos, right_center, line_color, 2)
    
    # Draw circles based on their current state
    if left_state == 2:  # Triggered
        cv2.circle(frame, left_center, radius, TRIGGERED_COLOR, CIRCLE_THICKNESS * 2)
        cv2.putText(frame, "TRIGGERED!", 
                    (left_center[0] - radius//2, left_center[1] + radius + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TRIGGERED_COLOR, 2)
    elif left_state == 1:  # Highlighted
        cv2.circle(frame, left_center, radius, HIGHLIGHT_COLOR, CIRCLE_THICKNESS * 2)
        cv2.putText(frame, "POINTED AT!", 
                    (left_center[0] - radius//2, left_center[1] + radius + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, HIGHLIGHT_COLOR, 2)
    else:  # Normal
        cv2.circle(frame, left_center, radius, CIRCLE_COLOR, CIRCLE_THICKNESS)
        
    if right_state == 2:  # Triggered
        cv2.circle(frame, right_center, radius, TRIGGERED_COLOR, CIRCLE_THICKNESS * 2)
        cv2.putText(frame, "TRIGGERED!", 
                    (right_center[0] - radius//2, right_center[1] + radius + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TRIGGERED_COLOR, 2)
    elif right_state == 1:  # Highlighted
        cv2.circle(frame, right_center, radius, HIGHLIGHT_COLOR, CIRCLE_THICKNESS * 2)
        cv2.putText(frame, "POINTED AT!", 
                    (right_center[0] - radius//2, right_center[1] + radius + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, HIGHLIGHT_COLOR, 2)
    else:  # Normal
        cv2.circle(frame, right_center, radius, CIRCLE_COLOR, CIRCLE_THICKNESS)

def process_frame_with_hands(frame: np.ndarray, overlay_data: Dict[str, Any]) -> List[HandLandmarkResult]:
    """Processes a frame to detect and visualize hands.
    
    This function combines hand detection and visualization.
    
    Args:
        frame: The input video frame
        overlay_data: Dictionary for text overlays
        
    Returns:
        List of HandLandmarkResult objects
    """
    # Detect hands
    hand_results = detect_hands(frame)
    
    # Process pointing gestures
    pointing_hands = []
    for i, hand_result in enumerate(hand_results):
        if is_pointing(hand_result.landmarks):
            direction = get_pointing_direction(hand_result.landmarks)
            pointing_hands.append((hand_result.landmarks, direction))
            
            # Check if trigger is being pulled
            trigger_status = ""
            text_color = (255, 255, 0)  # Default yellow for pointing
            
            if is_trigger_pulled(hand_result.landmarks):
                trigger_status = " (TRIGGER PULLED)"
                text_color = (0, 255, 0)  # Green for trigger pulled
                
            # Add text overlay indicating pointing detected
            overlay_data["texts"].append((
                f"Pointing detected: {hand_result.handedness} Hand{trigger_status}", 
                (10, 120 + i * 30),
                0.6,
                text_color
            ))
            
            # Add instructions for the user
            overlay_data["texts"].append((
                "Curl middle finger to pull trigger", 
                (frame.shape[1] - 350, frame.shape[0] - 20),
                0.6,
                (255, 255, 255)  # White text
            ))
    
    # Draw target circles and check if they're being pointed at
    draw_target_circles(frame, pointing_hands)
    
    # Draw hands on frame
    draw_hands(frame, hand_results, overlay_data)
    
    return hand_results