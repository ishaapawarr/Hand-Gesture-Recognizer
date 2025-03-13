import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

SIGN_MAP = {
    'Thumbs Up': lambda lm: lm[4].y < lm[3].y < lm[2].y and lm[8].y > lm[6].y,
    'Peace': lambda lm: lm[8].y < lm[6].y and lm[12].y < lm[10].y and lm[16].y > lm[14].y and lm[20].y > lm[18].y,
    'Okay': lambda lm: np.linalg.norm(np.array([lm[4].x, lm[4].y]) - np.array([lm[8].x, lm[8].y])) < 0.05,
    'Stop': lambda lm: len(lm) >= 21 and all(lm[i].y < lm[0].y for i in range(4, 21, 4)) and lm[8].y < lm[6].y
}

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            landmark_positions = landmarks.landmark

            detected_sign = "Unknown"
            for sign, check_fn in SIGN_MAP.items():
                if check_fn(landmark_positions):
                    detected_sign = sign
                    break

            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 70), (0, 0, 0), -1)
            cv2.putText(overlay, f"Sign: {detected_sign}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.imshow('Hand Gesture Recognizer', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
