from typing import Callable
from numpy import ndarray
import numpy as np
import mss
import cv2

def run_video(processing_pipeline: list[Callable[[np.ndarray], None]]) -> None:
    print("Starting video loop")
    with mss.mss() as sct:
        while True:
            try:
                # Capture the screen
                monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}  # You can adjust these values
                img = np.array(sct.grab(monitor))
                webcam_frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                # Type check for webcam_frame
                if isinstance(webcam_frame, ndarray):
                    for process in processing_pipeline:
                        process(webcam_frame)
            except KeyboardInterrupt:
                print("Program terminated")
                break