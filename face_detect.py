import cv2 as cv
from cv2 import imshow
import numpy as np

# Reading images from the folder
img = cv.imread('./Photos/group 1.jpg')
cv.imshow('group',img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# haar cascades are easy to use but cant be used for large projects

haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor = 1.1,minNeighbors=1)

print(f'no.of faces = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow('recct',img)



cv.waitKey(0)
cv.destroyAllWindows()