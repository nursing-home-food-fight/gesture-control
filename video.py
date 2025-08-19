from typing import Callable, Dict, List, Any, TypeVar
from numpy import ndarray
import numpy as np
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Generic return type for processing functions
T = TypeVar('T')

def run_video(processing_pipeline: List[Callable[[np.ndarray, Dict[str, Any]], Any]]) -> None:
    """
    Run video processing pipeline using webcam feed with text overlay capabilities.
    
    Args:
        processing_pipeline: List of functions that process frames and can add text to overlay
    """
    logger.debug("Starting video loop")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Error: Could not open webcam")
        return
    
    while True:
        try:
            # Capture frame from webcam
            ret, webcam_frame = cap.read()
            if not ret or not isinstance(webcam_frame, ndarray):
                logger.error("Error: Failed to capture frame from webcam")
                continue
                
            # Create overlay dictionary for processing functions to add text to
            overlay_data = {"texts": []}  # List of (text, position, font_scale, color)
            
            # Run processing pipeline
            for process in processing_pipeline:
                process(webcam_frame, overlay_data)
            
            # Draw all overlay texts
            for text_info in overlay_data.get("texts", []):
                text, position, font_scale, color = text_info
                cv2.putText(
                    webcam_frame, 
                    text, 
                    position, 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 
                    color, 
                    2
                )
            
            # Display the frame with overlays
            cv2.imshow("Webcam Feed", webcam_frame)
            
            # Exit on 'q' key press or if window was closed
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or cv2.getWindowProperty("Webcam Feed", cv2.WND_PROP_VISIBLE) < 1:
                logger.info("Program terminated by user")
                break
                
        except KeyboardInterrupt:
            logger.info("Program terminated")
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()