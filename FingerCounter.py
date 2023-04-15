__author__ = 'mkv-aql'
import cv2
import mediapipe as mp
import time
import os
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

#Get images
folderPath = "FingerImages"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}') #f'{folderPath}/{imPath}' is the path of the image ex: FingerImages/0.png
    overlayList.append(image) #Save images into a list

print("No. of element in the folder: ", len(overlayList)) #Print the number of images in the list

detector = htm.HandDetector(detectionCon=0.8)

tipIds = [4, 8, 12, 16, 20] #The ids of the landmarks of the tips of the fingers (thumb, index, middle, ring, pinky)
totalFingers = 0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, null = detector.findPosition(img, draw=False) #Because the findPosition function returns two values, the second one is not needed so it is assigned to null

    if len(lmList) != 0:

        fingers = []

        #Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:  #For the thumb, it moves across the x axis
            print("Finger is open")
            fingers.append(1)
        else:
            fingers.append(0)
            print("Finger is closed")

        #FIngers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]: #If the y coordinate of the tip of the finger is smaller than the y coordinate of the middle of the finger (2 is the y axis direction) (opencv orientation, the y axis is inverted, becasue the origin is at the top left corner)
                print("Finger is open")
                fingers.append(1)
            else:
                fingers.append(0)
                print("Finger is closed")

            '''
            if lmList[8][2] < lmList[6][2]: #If the tip of the index finger is bigger than the middle of the index finger (2 is the y axis direction) (opencv orientation, the y axis is inverted, becasue the origin is at the top left corner)
                print("Index finger is open")

            if lmList[8][2] > lmList[6][2]:
                print("Index finger is closed")
            '''

        print(fingers)
        #COUNTING THE NUMBER OF FINGERS THAT ARE OPEN, ADDING 1 FOR EACH FINGER THAT IS OPEN
        totalFingers = fingers.count(1)
        print(totalFingers)

    #Display the image
    h, w, c = overlayList[0].shape #Get the height, width and channel of the first image in the list
    #img[0:200, 0:200] = overlayList[0] #Display the first image at the top left corner of the screen (0:200, 0:200) is the pixel location of the image
    img[0:h, 0:w] = overlayList[totalFingers]

    cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 10)


    cv2.imshow("Finger", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break