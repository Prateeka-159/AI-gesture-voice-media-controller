import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
base_option = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_option,
    num_hands=1
)
landmarker = vision.HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)
prev_x, prev_y = 0, 0
movement_threshold = 20
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        h, w, _ = frame.shape
        wrist = hand[0]
        x = int(wrist.x*w)
        y = int(wrist.y*h)
        cv2.circle(frame, (x,y), 10, (0, 255, 0), -1)
        dx = x-prev_x
        dy = y-prev_y
        direction = ""
        if abs(dx) > movement_threshold:
            if dx>0:
                direction = "RIGHT"
            else:
                direction = "LEFT"
        if abs(dy) > movement_threshold:
            if dy>0:
                direction = "DOWN"
            else:
                direction = "UP"
        cv2.putText(frame, direction, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        prev_x, prev_y = x, y
    cv2.imshow("Hand movement detection", frame)
    if cv2.waitKey(1) & 0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
