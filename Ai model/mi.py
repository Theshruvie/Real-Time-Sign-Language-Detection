import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

SIGN_MAP_V2 = {
    "a": [4],
    "b": [8, 12, 16, 20],
    "c": [5, 9, 13, 17],
    "d": [8, 6, 10, 14],
    "e": [0, 5, 9, 13, 17],
    "f": [4, 8],
    "g": [8, 1],
    "h": [8, 12],
    "i": [20],
    "j": [20],
    "k": [8, 12, 2],
    "l": [8, 4],
    "m": [5, 9, 17, 0],
    "n": [5, 9, 13, 0],
    "o": [4, 5, 9, 13, 17],
    "p": [8, 12, 1],
    "q": [8, 4, 1],
    "r": [8, 10],
    "s": [0, 5, 9, 13, 17],
    "t": [4, 6, 9, 13, 17],
    "u": [8, 12],
    "v": [8, 12, 16],
    "w": [8, 12, 16, 0],
    "x": [7, 5],
    "y": [4, 20],
    "z": [8],
    "hello": [[8, 12, 16, 20], [4]],
    "shut_up": [[4], [8, 12, 16, 20]],
    "thumb_up": [8, 4],
    "peace": [12, 8],
    "okay": [8, 0],
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
            if np.linalg.norm(index_tip - thumb_tip) < 0.08 and index_tip[0] > wrist_landmark[0] and thumb_tip[0] > wrist_landmark[0]:
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

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    detected_sign_text = "none"
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detected_sign_text = detect_sign_v2(hand_landmarks)
            cv2.putText(frame, f"Sign: {detected_sign_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language Translator (Very Basic)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
