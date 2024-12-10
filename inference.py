# Import required libraries
import mediapipe as mp
import numpy as np
from PIL import Image
import io

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def detect_hand_landmarks(image_data):
    """
    Detect hand landmarks from an image.

    Args:
        image_data (bytes): Byte content of the input image.

    Returns:
        list: List of detected hand landmarks or None if no hand is detected.
    """
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image_np = np.array(image)

    # Process the image using MediaPipe
    results = hands.process(image_np)

    if results.multi_hand_landmarks:
        return results.multi_hand_landmarks[0].landmark
    return None

# Function to classify gestures
# Function to classify gestures
def classify_gesture(landmarks):
    """
    Classify gestures based on hand landmarks.

    Parameters:
        landmarks (list): List of hand landmarks detected by MediaPipe.

    Returns:
        str: Action associated with the gesture.
    """
    # Extract key landmarks
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Check for "Move Left" - Thumb and pinky extended
    if (
        thumb_tip.y < landmarks[3].y and  # Thumb extended
        index_tip.y < landmarks[6].y and  # Index extended
        middle_tip.y < landmarks[10].y and  # Middle extended
        ring_tip.y > landmarks[14].y and  # Ring folded
        pinky_tip.y > landmarks[17].y  # Pinky folded
    ):
        return "Move Left"

    # Check for "Move Right" - Thumb and index extended
    if (
        thumb_tip.y < landmarks[3].y and  # Thumb extended
        index_tip.y < landmarks[6].y and  # Index extended
        middle_tip.y > landmarks[10].y and  # Middle folded
        ring_tip.y > landmarks[14].y and  # Ring folded
        pinky_tip.y > landmarks[17].y  # Pinky folded
    ):
        return "Move Right"

    # Check for "Roll Down" - Thumb, index, and middle fingers extended
    if (
        thumb_tip.y < landmarks[3].y and  # Thumb extended
        pinky_tip.y < landmarks[17].y and  # Pinky extended
        index_tip.y > landmarks[6].y and  # Index folded
        middle_tip.y > landmarks[10].y and  # Middle folded
        ring_tip.y > landmarks[14].y  # Ring folded
    ):
        return "Roll Down"

    # Check for "Jump" - All fingers extended (open palm)
    if (
        thumb_tip.y < landmarks[3].y and
        index_tip.y < landmarks[6].y and
        middle_tip.y < landmarks[10].y and
        ring_tip.y < landmarks[14].y and
        pinky_tip.y < landmarks[17].y
    ):
        return "Jump"

    # Check for "Skateboard" - "ILY" gesture (Thumb, index, and pinky extended)
    if (
        thumb_tip.y < landmarks[3].y and  # Thumb extended
        index_tip.y < landmarks[6].y and  # Index extended
        pinky_tip.y < landmarks[17].y and  # Pinky extended
        middle_tip.y > landmarks[10].y and  # Middle folded
        ring_tip.y > landmarks[14].y  # Ring folded
    ):
        return "Skateboard"

    # Default action if no gesture is recognized
    return "No Action"
