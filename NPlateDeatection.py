import cv2
##########################
imgWidth = 540
imgHeight = 640
count =0
nPlateCascade = cv2.CascadeClassifier("Resources/haarcascade_russian_plate_number.xml")
##########################
cap = cv2.VideoCapture(0)
cap.set(3, imgHeight)
cap.set(4, imgWidth)
cap.set(10, 150)

while True:
    scanner, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 4)
    for(x, y,w, h) in numberPlates:
        area = w*h
        if area > 500:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,255), 2)
            cv2.putText(img, "Number Plate", (x, y-5), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,255), 2)
            imgROI= img[y:y+h, x:x+w]
            cv2.imshow("ROI", imgROI)

    cv2.imshow('Webcame', img)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('Resources/Scanned Plates/N_Plate' + str(count) + '.jpg', imgROI)
        cv2.rectangle(img, (x,y),(x+w,y+h), (0,255,0), 2, cv2.FILLED)
        cv2.putText(img,"Saved Sucessfully", (150,250), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)
        cv2.imshow("Result", img)
        count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


