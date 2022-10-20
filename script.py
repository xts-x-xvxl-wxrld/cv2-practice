import cv2
from datetime import datetime

first_frame = None
video = cv2.VideoCapture(0)
status_list = [None, None]
times = []
while True:
    check, frame = video.read()
    status = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    delta = cv2.absdiff(first_frame, gray)
    thresh_frame = cv2.threshold(delta, 50, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    (cnts, _) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status = 1

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    status_list.append(status)

    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    cv2.imshow('capture', gray)
    cv2.imshow('delta', delta)
    cv2.imshow('thresh', thresh_frame)
    cv2.imshow('color frame ', frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

print(times)
video.release()
cv2.destroyAllWindows()
