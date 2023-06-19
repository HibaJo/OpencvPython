import cv2
import numpy as np

'''This function only works with imgCanny asx input'''


def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area > 100:
            cv2.drawContours(imgContour, [cnt], -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, peri * 0.02, True)
            print(len(approx)) # Number of sides the shape has
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            if objCor == 3:
                objType = "Tri"
            elif objCor == 4:
                ratio= w/float(h)
                if ratio < 1.03 and ratio> 0.98:
                    objType = "square"
                else:
                    objType = "Rectangle"
            elif objCor > 4: objType = "Circle"
            else: objType = "None"

            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
            '''This output is affected by the resizing of the image'''
            cv2.putText(imgContour, objType, (x+w//2, y+h//2), cv2.FONT_HERSHEY_PLAIN, 0.7, (0,0,0),2)




img = cv2.imread("Resources/shapes.png")
# img = cv2.resize(img, (700, 600), 0)
imgContour = img.copy()

# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgBlur = cv2.GaussianBlur(imgGray, (7,7), 1)
imgCanny = cv2.Canny(img, 50, 50)

getContours(imgCanny)

# cv2.imshow("Original", img)
# cv2.imshow("Canny", imgCanny)
cv2.imshow("Contour", imgContour)
cv2.waitKey(0)
