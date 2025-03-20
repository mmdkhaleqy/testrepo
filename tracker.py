import cv2
import numpy as np
import winsound

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Not found webcam")
    exit()

sound_played = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("error")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if 'prev_frame' not in locals():
        prev_frame = gray
        continue

    diff = cv2.absdiff(prev_frame, gray)
    thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 1000:  
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        center_x = x + w // 2
        center_y = y + h // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        distance = 1000 / (w * h)
        cv2.putText(frame, f"Distance: {distance:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if distance < 0.5 and not sound_played:
            winsound.PlaySound('TrackerProject/alert.wav', winsound.SND_ASYNC)
            sound_played = True
        elif distance >= 0.5:
            sound_played = False

    prev_frame = gray.copy()

    cv2.imshow('Motion Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()