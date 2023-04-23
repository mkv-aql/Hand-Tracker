__author__ = 'mkv-aql'
'''
-Angle detection between 3 chosen landmarks. The angle is calculated by using the dot product of the 2 vectors.
-Angle detection between a chosen coordinate within a webcam frame and 2 chosen landmarks.
'''

import cv2
import HandTrackingModule as htm
import time
# For fps
pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)
# Calling the class
detector = htm.HandDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img, draw = True) # draw = False to get rid of the hand lines
    lmList, null = detector.findPosition(img, draw = True) # draw = False to avoid drawing the circles on the chosen landmarks, but still tracks location of the landmarks #Because the findPosition function returns two values, the second one is not needed so it is assigned to null
    # If there are landmarks detected, print the position of the tip of the index finger. Also to avoid index out of range error.
    if len(lmList) != 0:
        #print(lmList[7]) # Print the position of the tip of the index finger
        #print(detector.findAngle(img, 8, 7, 6, draw = True)) # Print the angle between the 3 chosen landmarks
        reference = detector.getPosition(img, pos = 'Bottom Center', draw=True)
        print(detector.getPosition(img, pos = 'Bottom Center', draw=True))
        print(detector.findAngle(img, reference, 0, 5, draw = True)) # Print the angle between the 3 chosen landmarks and the reference coordinate


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