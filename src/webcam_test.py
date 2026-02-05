import cv2
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, "Wedcamera Test", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destryAllWindows()