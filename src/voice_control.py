import speech_recognition as sr
import pyautogui
import pygetwindow as gw
import time

def focus_browser():
    windows = gw.getAllTitles()
    for title in windows:
        if "YouTube" in title or "Chrome" in title or "Edge" in title:
            try:
                win = gw.getWindowsWithTitle(title)[0]
                win.activate()
                time.sleep(0.5)
                return True
            except:
                pass
    return False

recognizer = sr.Recognizer()
mic = sr.Microphone()
print("Voice control started. Say 'play' or 'pause'.")

while True:
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = recognizer.listen(source)

        command = recognizer.recognize_google(audio).lower()
        print("You said:", command)

        if "pause" in command or "play" in command:
            if focus_browser():
                pyautogui.press("playpause")
                print("Toggled Play/Pause")
            else:
                print("Browser not found")

    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError:
        print("Internet error")

