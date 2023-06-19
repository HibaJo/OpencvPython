import cv2

faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")

'''Face Detection from pictures'''
img = cv2.imread('Resources/faces.jpeg')
img = cv2.resize(img, (700, 500))
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(imgGray,1.1,4)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h),(0,0,255), 2)

cv2.imshow("Image Face Detection", img)
cv2.waitKey(0)

'''Face detection from an open camera'''
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    #camera to be converted to Gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)
    for (x,y, w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow('Camera Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
