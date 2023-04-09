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
    lmList = detector.findPosition(img, draw = True) # draw = False to avoid drawing the circles on the chosen landmarks, but still tracks location of the landmarks
    # If there are landmarks detected, print the position of the tip of the index finger. Also to avoid index out of range error.
    if len(lmList) != 0:
        print(lmList[8]) # Print the position of the tip of the index finger

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