import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import speech_recognition as sr
import threading
import time

pyautogui.FAILSAFE = True

# voice recognition
def voice_control():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    while True:
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)
            command = recognizer.recognize_google(audio).lower()
            if "pause" in command or "play" in command:
                pyautogui.press("playpause")
                print("Voice command:", command)
        except:
            pass

# voice thread
threading.Thread(target=voice_control, daemon=True).start()

# hand tracking and movement
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
prev_x, prev_y = 0, 0
movement_threshold = 20
cooldown = 0.4
last_action_time = time.time()

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        h, w, _ = frame.shape
        wrist = hand[0]
        x = int(wrist.x * w)
        y = int(wrist.y * h)
        cv2.circle(frame, (x, y), 10, (0,255,0), -1)
        dx = x - prev_x
        dy = y - prev_y
        current_time = time.time()
        direction = ""

        if abs(dx) > movement_threshold or abs(dy) > movement_threshold:
            if current_time - last_action_time > cooldown:
                if abs(dx) > abs(dy):
                    if dx > 0:
                        pyautogui.hscroll(-200)
                        direction = "RIGHT"
                    else:
                        pyautogui.hscroll(200)
                        direction = "LEFT"
                else:
                    if dy > 0:
                        pyautogui.scroll(-200)
                        direction = "DOWN"
                    else:
                        pyautogui.scroll(200)
                        direction = "UP"
                last_action_time = current_time
        cv2.putText(frame, direction, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        prev_x, prev_y = x, y
    cv2.imshow("Gesture + Voice Controller", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
