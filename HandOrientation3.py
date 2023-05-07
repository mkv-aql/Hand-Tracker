__author__ = 'mkv-aql'
'''
Hand orientation.
-Getting the 3 chosen landmarks of the hand, which are 0, 5, 17.
-Convert the landmarks to a numpy array.
-define the vectors based on the landmarks. (eg. from 0 to 17, from 17 to 5)
'''

import cv2
import HandTrackingModule as htm
import time
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = htm.HandDetector(detectionCon=0.8)
pTime = 0
cTime = 0


while True:
    success, img = cap.read()
    img = detector.findHands(img, draw = False) # draw = False to get rid of the hand lines
    lmList, null = detector.findPosition(img, draw = True) # draw = False to avoid drawing the circles on the chosen landmarks, but still tracks location of the landmarks
    # If there are landmarks detected, print the position of the tip of the index finger. Also to avoid index out of range error.
    if len(lmList) != 0:
        print(lmList[0]) # Print the position of the tip of the palm




    # For fps calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    # Draw fps on image
    cv2.putText(img, str(int(fps)), (10, 35), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cv2.imshow("Image", img)
    # waitKey with q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break