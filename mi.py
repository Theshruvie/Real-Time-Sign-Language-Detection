import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Adjusted Sign Map (more distinct for alphabets and including other signs)
SIGN_MAP_V2 = {
    "a": [4],          # Thumb TIP extended
    "b": [8, 12, 16, 20], # All finger TIPS extended
    "c": [5, 9, 13, 17], # All finger MCPs extended (approximation of curved)
    "d": [8, 6, 10, 14], # Index TIP & PIP extended, others curled (approx)
    "e": [0, 5, 9, 13, 17], # Wrist and all finger MCPs extended (approx curled tips)
    "f": [4, 8],          # Thumb TIP and Index TIP touching
    "g": [8, 1],          # Index TIP and Thumb CMC extended (sideways approx)
    "h": [8, 12],         # Index and Middle TIP extended
    "i": [20],         # Pinky TIP extended
    "j": [20],         # Pinky TIP extended (movement is key)
    "k": [8, 12, 2],     # Index & Middle TIP extended, Thumb MCP extended
    "l": [8, 4],          # Index TIP and Thumb TIP at angle
    "m": [5, 9, 17, 0],   # MCPs of I, M, P extended, Wrist (approx thumb tuck)
    "n": [5, 9, 13, 0],   # MCPs of I, M, R extended, Wrist (approx thumb tuck)
    "o": [4, 5, 9, 13, 17], # Thumb TIP and all MCPs extended (approx circle)
    "p": [8, 12, 1],     # Index & Middle TIP down, Thumb CMC extended
    "q": [8, 4, 1],      # Index TIP down, Thumb TIP & CMC extended
    "r": [8, 10],         # Index TIP & Middle PIP extended
    "s": [0, 5, 9, 13, 17], # Wrist & all MCPs (very rough fist)
    "t": [4, 6, 9, 13, 17], # Thumb TIP & Index PIP extended, others MCP
    "u": [8, 12],         # Index & Middle TIP extended (pointing up)
    "v": [8, 12, 16],      # Index, Middle TIPS extended
    "w": [8, 12, 16, 0],  # Index, Middle, Ring TIPS extended, Wrist
    "x": [7, 5],          # Index DIP & MCP extended (hook approx)
    "y": [4, 20],         # Thumb TIP and Pinky TIP extended
    "z": [8],             # Index TIP extended (movement is key)
    "hello": [[8, 12, 16, 20], [4]], # Waving motion (represented by all fingers and thumb extended)
    "shut_up": [[4], [8, 12, 16, 20]], # Thumb extended, others closed (rough)
    "thumb_up": [8, 4],    # Tip of index finger above thumb tip (very rough)
    "peace": [12, 8],      # Tip of middle and index extended
    "okay": [8, 0],       # Index tip and wrist (very rough circle approximation)
    "none": []
}

FINGER_LANDMARK_INDICES = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20
}

def get_finger_extended(finger_name, landmarks):
    tip_index = FINGER_LANDMARK_INDICES[finger_name]
    mcp_index = tip_index - 1
    if finger_name == "thumb":
        dip_index = 2
        return landmarks[tip_index].x > landmarks[dip_index].x
    else:
        pip_index = tip_index - 2
        return landmarks[tip_index].y < landmarks[pip_index].y

def detect_sign_v2(hand_landmarks):
    """
    A VERY basic and HIGHLY INACCURATE function to detect signs (including alphabets)
    based on landmark positions, using the adjusted SIGN_MAP_V2.
    """
    if not hand_landmarks:
        return "none"

    landmarks = hand_landmarks.landmark
    extended_finger_indices = []

    if get_finger_extended("thumb", landmarks):
        extended_finger_indices.append(FINGER_LANDMARK_INDICES["thumb"])
    if get_finger_extended("index", landmarks):
        extended_finger_indices.append(FINGER_LANDMARK_INDICES["index"])
    if get_finger_extended("middle", landmarks):
        extended_finger_indices.append(FINGER_LANDMARK_INDICES["middle"])
    if get_finger_extended("ring", landmarks):
        extended_finger_indices.append(FINGER_LANDMARK_INDICES["ring"])
    if get_finger_extended("pinky", landmarks):
        extended_finger_indices.append(FINGER_LANDMARK_INDICES["pinky"])

    sorted_extended = sorted(extended_finger_indices)

    for sign, pattern in SIGN_MAP_V2.items():
        if isinstance(pattern, list) and sign not in ["hello", "shut_up", "thumb_up", "peace", "okay"]:
            if sorted(pattern) == sorted_extended:
                return sign
        elif sign == "thumb_up":
            index_tip = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
            thumb_tip = np.array([landmarks[mp_hands.HandLandmark.THUMB_TIP].x, landmarks[mp_hands.HandLandmark.THUMB_TIP].y])
            if index_tip[1] < thumb_tip[1] and np.linalg.norm(index_tip - thumb_tip) < 0.1:
                return sign
        elif sign == "peace":
            index_tip = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
            middle_tip = np.array([landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y])
            ring_tip = np.array([landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y])
            pinky_tip = np.array([landmarks[mp_hands.HandLandmark.PINKY_TIP].x, landmarks[mp_hands.HandLandmark.PINKY_TIP].y])
            wrist = np.array([landmarks[mp_hands.HandLandmark.WRIST].x, landmarks[mp_hands.HandLandmark.WRIST].y])
            if np.linalg.norm(index_tip - middle_tip) < 0.05 and index_tip[1] < wrist[1] and middle_tip[1] < wrist[1] and ring_tip[1] > wrist[1] and pinky_tip[1] > wrist[1]:
                return sign
        elif sign == "okay":
            index_tip = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
            thumb_tip = np.array([landmarks[mp_hands.HandLandmark.THUMB_TIP].x, landmarks[mp_hands.HandLandmark.THUMB_TIP].y])
            wrist_landmark = np.array([landmarks[mp_hands.HandLandmark.WRIST].x, landmarks[mp_hands.HandLandmark.WRIST].y])
            if np.linalg.norm(index_tip - thumb_tip) < 0.08 and index_tip[0] > wrist_landmark[0] and thumb_tip[0] > wrist_landmark[0]: # Very rough approximation
                return sign
        elif isinstance(pattern, list) and len(pattern) == 2 and isinstance(pattern[0], list) and isinstance(pattern[1], list):
            extended_all = sorted_extended == sorted([4, 8, 12, 16, 20])
            if sign == "hello" and extended_all:
                return sign
            thumb_extended = 4 in extended_finger_indices
            other_fingers_not_extended = all(idx not in extended_finger_indices for idx in [8, 12, 16, 20])
            if sign == "shut_up" and thumb_extended and other_fingers_not_extended:
                return sign

    return "none"

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame horizontally for a more natural self-view
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    detected_sign_text = "none"
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detected_sign_text = detect_sign_v2(hand_landmarks)
            cv2.putText(frame, f"Sign: {detected_sign_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Sign Language Translator (Very Basic)", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
