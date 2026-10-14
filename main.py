import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Gesture Mapping
gesture_map = {
    "Good": [(4, 3, "above"), (8, 6, "below"), (12, 10, "below"), (16, 14, "below"), (20, 18, "below")],
    "OK": [(4, 8, "close"), (12, 10, "above"), (16, 14, "above"), (20, 18, "above")],
    "Peace": [(8, 6, "above"), (12, 10, "above"), (16, 14, "below"), (20, 18, "below")]
}

last_detected_gesture = None


def detect_gesture(landmarks):
    for gesture, conditions in gesture_map.items():
        if all(check_condition(landmarks, *cond) for cond in conditions):
            return gesture
    return "Unknown"


def check_condition(landmarks, point1, point2, relation):
    if relation == "above":
        return landmarks[point1].y < landmarks[point2].y
    elif relation == "below":
        return landmarks[point1].y > landmarks[point2].y
    elif relation == "close":
        return abs(landmarks[point1].x - landmarks[point2].x) < 0.05 and abs(
            landmarks[point1].y - landmarks[point2].y) < 0.05
    return False


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detected_gesture = detect_gesture(hand_landmarks.landmark)
            if detected_gesture != last_detected_gesture:
                print(detected_gesture)  # Output to terminal only when it changes
                last_detected_gesture = detected_gesture
            cv2.putText(frame, detected_gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
