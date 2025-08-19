from typing import Any
from typing import Dict, List
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
    
    # Draw hands on frame
    draw_hands(frame, hand_results, overlay_data)
    
    return hand_results