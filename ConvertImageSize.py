__author__ = 'mkv-aql'

#Convert image size to 200x200
import cv2
import mediapipe as mp
import time
import os

#get images
folderPath = "FingerImages"
myList = os.listdir(folderPath)

#convert images to 200x200
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}') #f'{folderPath}/{imPath}' is the path of the image ex: FingerImages/0.png
    image = cv2.resize(image, (200, 200))
    cv2.imwrite(f'{folderPath}/{imPath}', image)

