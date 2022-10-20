import cv2


front_cascade = cv2.CascadeClassifier('head_front_cascade.xml')
path = 'Before.png'

image = cv2.imread(path, 1)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, [5,5], cv2.BORDER_DEFAULT)

faces = front_cascade.detectMultiScale(gray,
                                       scaleFactor=1.1,
                                       minNeighbors=3)

for x, y, w, h in faces:
    #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 2)
    extract = image[y:y+h, x:x+w]
    cv2.imwrite('scalp.jpg', extract)

