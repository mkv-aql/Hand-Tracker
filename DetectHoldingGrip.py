__author__ = 'mkv-aql'
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
    img = detector.findHands(img, draw = False) # draw = False to get rid of the hand lines
    lmList, null = detector.findPosition(img, draw = False) # draw = False to avoid drawing the circles on the chosen landmarks, but still tracks location of the landmarks #Because the findPosition function returns two values, the second one is not needed so it is assigned to null
    # If there are landmarks detected, print the position of the tip of the index finger. Also to avoid index out of range error.
    if len(lmList) != 0:
        print(lmList[0])
        #point = detector.drawBoundariesTest(img, 0, draw = True)
        topLeft = (213, 160)
        bottomRight = (426, 320)
        point = detector.withinBoundaries(img, 5, topLeft, bottomRight, draw = True)
        print("Point is within boundary: ", point)
        angle1 = detector.findAngle(img, 0, 5, 6, draw=True)
        angle2 = detector.findAngle(img, 5, 6, 7, draw=True)
        if ((angle1 + angle2) < 300) and (point == True):
            print("Holding Grip")
        else:
            print("Not Holding Grip")


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