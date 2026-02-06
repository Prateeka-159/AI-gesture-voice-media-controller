import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import speech_recognition as sr
import threading
import time
import math

pyautogui.FAILSAFE = True

# ---------------- Voice control thread ----------------
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

# ---------------- Hand model ----------------
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

mode = "LOCKED"
smooth_x, smooth_y = 0, 0
alpha = 0.2

last_scroll_time = 0
scroll_cooldown = 0.6

last_zoom_time = 0
zoom_cooldown = 0.8

prev_pinch_distance = 0

prev_state = "OPEN"
lock_state = True

# Detect fingers up
def fingers_up(hand, h, w):
    thumb_up = int(hand[4].x*w) > int(hand[3].x*w)
    index_up = int(hand[8].y*h) < int(hand[6].y*h)
    middle_up = int(hand[12].y*h) < int(hand[10].y*h)
    ring_up = int(hand[16].y*h) < int(hand[14].y*h)
    pinky_up = int(hand[20].y*h) < int(hand[18].y*h)
    return thumb_up, index_up, middle_up, ring_up, pinky_up

# Detect closed fist reliably
def is_fist(hand, h, w):
    thumb_closed = int(hand[4].x*w) < int(hand[3].x*w)
    index_closed = int(hand[8].y*h) > int(hand[6].y*h)
    middle_closed = int(hand[12].y*h) > int(hand[10].y*h)
    ring_closed = int(hand[16].y*h) > int(hand[14].y*h)
    pinky_closed = int(hand[20].y*h) > int(hand[18].y*h)
    return thumb_closed and index_closed and middle_closed and ring_closed and pinky_closed

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        h,w,_ = frame.shape

        # Draw ALL 21 landmarks as green dots
        for lm in hand:
            px = int(lm.x * w)
            py = int(lm.y * h)
            cv2.circle(frame, (px, py), 5, (0,255,0), -1)

        thumb,index,middle,ring,pinky = fingers_up(hand,h,w)
        fist = is_fist(hand,h,w)

        # -------- LOCK / UNLOCK using fist -> open palm --------
        if prev_state == "FIST" and not fist:
            lock_state = not lock_state
            print("LOCKED" if lock_state else "UNLOCKED")

        prev_state = "FIST" if fist else "OPEN"

        if lock_state:
            cv2.putText(frame,"LOCKED",(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.imshow("Gesture Controller", frame)
            if cv2.waitKey(1)&0xFF==27:
                break
            continue

        # -------- ZOOM MODE (L shape only) --------
        if thumb and index and not middle and not ring and not pinky:
            mode = "ZOOM"

            thumb_tip = hand[4]
            index_tip = hand[8]
            tx,ty = int(thumb_tip.x*w), int(thumb_tip.y*h)
            ix,iy = int(index_tip.x*w), int(index_tip.y*h)
            pinch_distance = math.hypot(tx-ix, ty-iy)

            if prev_pinch_distance != 0 and time.time()-last_zoom_time > zoom_cooldown:
                diff = pinch_distance - prev_pinch_distance
                if abs(diff) > 30:
                    pyautogui.hotkey("ctrl","+" if diff>0 else "-")
                    last_zoom_time = time.time()

            prev_pinch_distance = pinch_distance

        # -------- CURSOR MODE --------
        elif index and not middle and not thumb:
            mode = "CURSOR"

            x = int(hand[8].x * w)
            y = int(hand[8].y * h)

            smooth_x = alpha*x + (1-alpha)*smooth_x
            smooth_y = alpha*y + (1-alpha)*smooth_y

            screen_w, screen_h = pyautogui.size()
            pyautogui.moveTo(screen_w*(smooth_x/w), screen_h*(smooth_y/h), duration=0)

        # -------- SCROLL MODE --------
        elif index and middle and not thumb:
            mode = "SCROLL"

            if time.time()-last_scroll_time > scroll_cooldown:
                y = int(hand[8].y * h)
                if y < h*0.4:
                    pyautogui.scroll(400)
                    last_scroll_time = time.time()
                elif y > h*0.6:
                    pyautogui.scroll(-400)
                    last_scroll_time = time.time()

        else:
            mode = "IDLE"

        cv2.putText(frame, f"MODE: {mode}", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

    cv2.imshow("Gesture Controller", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
