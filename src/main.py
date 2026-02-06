import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import speech_recognition as sr
import threading
import time

pyautogui.FAILSAFE = True

# ---------------- VOICE CONTROL ----------------
def voice_control():
    r = sr.Recognizer()
    mic = sr.Microphone()
    while True:
        try:
            with mic as source:
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source)
            cmd = r.recognize_google(audio).lower()
            if "pause" in cmd or "play" in cmd:
                pyautogui.press("playpause")
        except:
            pass

threading.Thread(target=voice_control, daemon=True).start()

# ---------------- MEDIAPIPE HAND MODEL ----------------
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

prev_y = 0
scroll_speed = 40

# ---------- Robust fist detection ----------
def is_fist(hand, h):
    fingertip_ids = [8,12,16,20]
    knuckle_ids   = [6,10,14,18]

    tip_avg = sum([hand[i].y for i in fingertip_ids]) / 4
    knuckle_avg = sum([hand[i].y for i in knuckle_ids]) / 4

    # fingertips lower than knuckles → fist
    return tip_avg > knuckle_avg

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        h, w, _ = frame.shape

        # draw all landmarks
        for lm in hand:
            px = int(lm.x * w)
            py = int(lm.y * h)
            cv2.circle(frame, (px, py), 4, (0,255,0), -1)

        wrist = hand[0]
        y = int(wrist.y * h)

        if is_fist(hand, h):
            dy = y - prev_y

            if dy < -4:
                pyautogui.scroll(scroll_speed)
                cv2.putText(frame,"FIST SCROLL UP",(30,50),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            elif dy > 4:
                pyautogui.scroll(-scroll_speed)
                cv2.putText(frame,"FIST SCROLL DOWN",(30,50),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,"OPEN HAND = STOP",(30,50),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        prev_y = y

    cv2.imshow("Fist Scroll + Voice Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
