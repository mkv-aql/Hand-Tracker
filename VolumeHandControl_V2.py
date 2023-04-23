__author__ = 'mkv-aql'
'''
-Volume Control with index finger and thumb, and pinky finger to set (locked) the volume.
-Drawing on the hand changed to True (draw=True), to draw the bounding box around the hand
'''
import cv2
import mediapipe as mp
import time
import numpy as np
import HandTrackingModule as htm
import math

#PARAMETERS FOR THE OUTPUT VIDEO RESOLUTION
wCam, hCam = 640, 480
    #Set up the video capture and set the resolution
cap = cv2.VideoCapture(0)
cap.set(3, wCam)#3 corresponds to 'CV_CAP_PROP_FRAME_WIDTH', then itsets the width of the video
cap.set(4, hCam) #4 corresponds to 'CV_CAP_PROP_FRAME_HEIGHT', then it sets the height of the video

#for fps
pTime = 0

#Calling the class
detector = htm.HandDetector(detectionCon=0.8) #detectionCon=0.8 to increase the accuracy of the hand detection (Find true image of hand)

#VOLUME CONTROL LIBRARY, PARAMETERS, AND INITIALIZATION (from https://github.com/AndreMiras/pycaw)
    #import
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    #initialization
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
    #parameters
volRange = volume.GetVolumeRange()
volume.SetMasterVolumeLevel(0.0, None)
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400 #For the volume bar (400 is the max value of the volume bar) (the pixel location)
volPer = 0 #For the volume bar (the percentage of the volume bar)

#BOUNDING BOX PARAMETERS
area = 0 #bounding box area

colorVol = (255, 0, 0) #Color of the volume

while True:
    success, img = cap.read()

    #FIND HAND LANDMARKS
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=True) #draw=True to draw the default circles on the chosen landmarks, and will override the drawing below. And receive the bounding box of the hand coordinates
    if len(lmList) != 0:

        #FILTER BASED ON SIZE (so that there is a distance tolereance from the webcam to hand, so that further away hands are not considered a 'condition')
            #Finding the bounding box of the hand
        wB, hB = bbox[2] - bbox[0], bbox[3] - bbox[1] #wB is the width of the bounding box, hB is the height of the bounding box
        area = wB * hB // 100 #area is the area of the bounding box
        print(area)
        #print(bbox) #Prints the coordinates of the bounding box (xmin, ymin, xmax, ymax)

        if 250 < area < 1100: #250 is the minimum area, 1100 is the maximum area (the area of the bounding box), in other words it is the distance of the hand from the webcam tolerence
            print("Area is within the range")
            #FIND THE DISTANCE BETWEEN THE TIP OF THE THUMB AND THE TIP OF THE INDEX FINGER
            length, img, lineInfo = detector.findDistance(img, 4, 8, draw=True, drawLength=True) #4 is the tip of the thumb, 8 is the tip of the index finger

            #CONVERT THE LENGTH TO VOLUME

            if length < 50:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 7, (0, 255, 0), cv2.FILLED)



            # VOLUME BAR LIMIT SET UP, #Hand range 50 - 250, #Volume range -96 - 0
            volBar = np.interp(length, [50, 250],
                                   [400, 150])  # Interpolate the volume between the min and max volume
            volPer = np.interp(length, [50, 250],
                                   [0, 100])  # Interpolate the volume between the min and max volume for the percentage
            vol = np.interp(length, [50, 250], [minVol, maxVol]) #Interpolate the volume between the min and max volume
                # Reduce resolution to make it smoother
            smoothness = 2
            volPer = smoothness * round(volPer / smoothness)
            print("dB: ", vol)



            #CHECK WHICH FINGERS ARE UP
            fingers = detector.fingersUp()
            print(fingers)
                # If the pinky finger is down, set the volume
            if fingers[4] == False:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 7, (255, 255, 0), cv2.FILLED)
                volume.SetMasterVolumeLevelScalar(volPer/100, None) #Set the volume to the interpolated value
                colorVol = (255, 255, 0) #change color of the volume
            else:
                colorVol = (255, 0, 0) #change color of the volume back to blue



    #DRAWING A VOLUME BAR
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 250, 0), 2)

    #DRAW WINDOWS VOLUME
    cVol = int(volume.GetMasterVolumeLevelScalar() * 100)
    cv2.putText(img, f'CVolume: {int(cVol)} %', (400, 50), cv2.FONT_HERSHEY_COMPLEX, 1, colorVol, 2)



    #FRAME RATE CALCULATION
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    #DRAW FPS ON IMAGE
    cv2.putText(img, f'fps: {str(int(fps))}', (10, 35), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)


    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



