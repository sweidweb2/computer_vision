import cv2 as cv
import imutils
import numpy as np
import matplotlib.pyplot as plt


def d(image):
    cv.imshow("new",image)
    cv.waitKey(0)

def p(image):
    plt.imshow(image,cmap='gray')
    plt.show()

# load predefined classifiers
# Load the Haar Cascade classifiers from OpenCV's data
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade  = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

if face_cascade.empty():
    print("Failed to load face cascade")
if eye_cascade.empty():
    print("Failed to load eye cascade")


img=cv.imread('DATA/Nadia_Murad.jpg')

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# d(img)

faces=face_cascade.detectMultiScale(gray,1.3,5)

for(x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray=gray[y:y+h,x:x+w]
    roi_color=img[y:y+h,x:x+w]

    eyes=eye_cascade.detectMultiScale(roi_gray,1.8)

    for (ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)


cv.imshow("eyess and faces",img)
cv.waitKey(0)
cv.destroyAllWindows()


