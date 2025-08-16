import cv2
import mediapipe as mp
from typing import Any, Final, Tuple
import numpy as np
from numpy.typing import NDArray

mp_hands: Any = mp.solutions.hands
mp_drawing: Any = mp.solutions.drawing_utils
mp_styles: Any = mp.solutions.drawing_styles

# --- Config ---
FRAME_WIDTH: Final[int] = 1280
FRAME_HEIGHT: Final[int] = 720
CIRCLE_COLOR: Final[Tuple[int, int, int]] = (0, 0, 255)   # red


# --- Helpers ---

def draw_circle(img: NDArray[np.uint8]) -> None:
    h, w = img.shape[:2]
    radius = int(min(w, h) * 0.18)  # larger radius
    thickness = 18  # thicker
    cv2.circle(img, (w // 2, h // 2), radius, CIRCLE_COLOR, thickness=thickness)

def is_fist(hand_lms: Any) -> bool:
    # Simple fist detection: all fingertips close to palm
    # Indices: 0=wrist, 4=thumb_tip, 8=index_tip, 12=middle_tip, 16=ring_tip, 20=pinky_tip
    tips = [4, 8, 12, 16, 20]
    palm = hand_lms.landmark[0]
    closed = 0
    for tip_idx in tips:
        tip = hand_lms.landmark[tip_idx]
        dist = ((tip.x - palm.x) ** 2 + (tip.y - palm.y) ** 2) ** 0.5
        if dist < 0.18:
            closed += 1
    return closed >= 3

cv2.destroyAllWindows()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
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
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )
                if is_fist(hand_lms):
                    fist_detected = True
        if fist_detected:
            draw_circle(frame)
        cv2.putText(frame, 'Make a closed fist with either hand to draw a circle. Press x to exit.', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)
        cv2.imshow('Gesture Circle', frame.astype(np.uint8))
        if (cv2.waitKey(1) & 0xFF) == ord('x'):
            break
cap.release()
cv2.destroyAllWindows()
print('Gesture demo stopped.')
