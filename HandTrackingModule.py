__author__ = 'mkv-aql'
'''
Update 15.4.2023: 
-Added fingersUp function (in Class) to detect how many fingers are up
-findPosition function editted: to fund the maximum x and y and minmum x and y values of the landmarks. (This is to find the bounding box of the hand)
-Added findDistance function to find the distance between two landmarks

Update 23.4.2023:
-Added the left and right hand detection (in findPosition Class)
'''
import cv2
import mediapipe as mp
import time
import numpy as np
import math
from matplotlib import pyplot as plt


class HandDetector():
    # Attributes
    tipIds = [4, 8, 12, 16, 20]  # The ids of the landmarks of the tips of the fingers (thumb, index, middle, ring, pinky)

    #Constructor / initializer
    def __init__(self, mode = False, maxHands = 2, modelC = 1, detectionCon = 0.5, trackCon = 0.5): #, static_image_mode=False, max_num_hands=2, model_complexity = 1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        #Constructor
        self.mode = mode
        self.maxHands = maxHands
        self.modelC = modelC # model_complexity must be added (new in mediapipe 0.8.3)
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelC, self.detectionCon, self.trackCon)  # static_image_mode=False, max_num_hands=2, model_complexity = 1, min_detection_confidence=0.5, min_tracking_confidence=0.5), If static_image_mode=True, it will continously detect hands (slower performance)
        self.mpDraw = mp.solutions.drawing_utils



    #Detection function / methods
    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # Extracting the landmarks from results
        # print(results.multi_hand_landmarks)
        # If there are hands detected, for each hand (handLms), draw the landmarks and show id no. with its location.
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    if draw:
                        self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    # Find position of a specific landmark (if handNo = 0, it will find the position of the first hand detected)
    def findPosition(self, img, handNo = 0, draw = True):
        self.lmList = []
            #List for the x min/max and y min/max coordinates of the landmarks (15.5.2023)
        xList = []
        yList = []
        bbox = 0, 0, 0, 0

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                lr = self.results.multi_handedness[handNo].classification[handNo].label # Left or right hand (23.4.2023)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                    #Append cx, cy values into the list (15.4.2023)
                xList.append(cx)
                yList.append(cy)

                #print(id, cx, cy)  # id is the landmark id no. and lm is the landmark location. id = 0 is the palm of the hand, id = 4 is thw tip of the thumb
                self.lmList.append([id, cx, cy, lr])
                if draw:
                    #Highlight the landmark with a circle of id no. 4 (tip of the thumb)
                    if id == 4:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)  # Draw a circle on the tip of the thumb
                    #Highlight the landmark with a circle of id no. 8 (tip of the index finger)
                    if id == 8:
                        cv2.circle(img, (cx, cy), 5, (255, 255, 0), cv2.FILLED)  # Draw a circle on the tip of the thumb
                    # Highlight the landmark with a circle of id no. 0 (palm)
                    if id == 0:
                        cv2.circle(img, (cx, cy), 5, (255, 255, 255), cv2.FILLED)  # Draw a circle on the tip of the thumb
                        cv2.putText(img, str(lr), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)  # Draw the left or right hand on the palm (23.4.2023)

            #FIND THE MINIMUM AND MAXIMUM X AND Y VALUES OF THE LANDMARKS (15.4.2023)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax # Bounding box of the hand
            #DRAW THE BOUNDING BOX (15.4.2023)
            if draw:
                cv2.rectangle(img, (bbox[0]-20, bbox[1]-20), (bbox[2]+20, bbox[3]+20), (0, 255, 0), 2) # Draw a rectangle around the hand (takes in the top left and bottom right coordinates of the rectangle). -20 and +20 is to make the bounding box bigger

        return self.lmList, bbox

    # Find the number of fingers up (0 or 1) (0 = finger is down, 1 = finger is up)
    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers # Return a list of 5 elements (0 or 1) to indicate if the finger is up or not

    # Find the distance between two landmarks (p1 and p2 are the landmark id no.)
    def findDistance(self, img, p1, p2, draw =True, drawLength = True):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]  # x1, y1 are the coordinates of the tip of the thumb finger
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]  # x2, y2 are the coordinates of the tip of the index finger
        cx, cy = (x1 + x2) // 2, (
                    y1 + y2) // 2  # cx, cy are the coordinates of the center of the line between the tip of the thumb and the tip of the index finger (center of the line between the two fingers)

        if draw:
            # Draw a circle on the tip of the thumb and the tip of the index finger
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)  # Draw a circle on the tip of the thumb
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)  # Draw a circle on the tip of the index finger
            # Draw a line between the tip of the thumb and the tip of the index finger
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            # Draw a circle on the center of the line between the tip of the thumb and the tip of the index finger
            cv2.circle(img, (cx, cy), 7, (0, 0, 255), cv2.FILLED)

        if drawLength:
            # Calculate the length of the line between the 2 points
            length = np.hypot(x2 - x1, y2 - y1)
            print("Length: ", int(length))

        return length, img, [x1, y1, x2, y2, cx, cy]

    def findAngle(self, img, p1, p2, p3, handNo = 0, draw = True):
        jointList = [p1, p2, p3]
        test = []
        angle = 0
        #coordinates x y
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        x3, y3 = self.lmList[p3][1], self.lmList[p3][2]

        if self.results.multi_hand_landmarks:

            for myHand in self.results.multi_hand_landmarks:

                for joints in jointList:
                    a = np.array([x1, y1], np.int32)
                    b = np.array([x2, y2], np.int32)
                    c = np.array([x3, y3], np.int32)
                    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                    angle = np.abs(radians*180/np.pi)
                    if angle > 180:
                        angle = 360 - angle

        if draw:
            cv2.putText(img, str(p2), (x2, y2), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            cv2.putText(img, str(round(angle, 2)), (x2, y2+10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
        return angle


def main():
    # For fps
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0)
    # Calling the class
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        # If there are landmarks detected, print the position of the tip of the index finger. Also to avoid index out of range error.
        if len(lmList) != 0:
            print(lmList[8]) # Print the position of the tip of the index finger

        # For fps calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        # Draw fps on image
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cv2.imshow("Image", img)
        # waitKey with q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()