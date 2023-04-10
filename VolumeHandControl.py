__author__ = 'mkv-aql'
import cv2
import mediapipe as mp
import time
import numpy as np
import HandTrackingModule as htm
import math

#Parameters
wCam, hCam = 640, 480


cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

#for fps
pTime = 0

#Calling the class
detector = htm.HandDetector(detectionCon=0.8) #detectionCon=0.8 to increase the accuracy of the hand detection (Find true image of hand)

#Volume Control library, parameters, and initialization (from https://github.com/AndreMiras/pycaw)
    #import
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    #initialization
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
    #parameters
'''
#volume.GetMute()
#volume.GetMasterVolumeLevel() # Get the current volume in dB
#volume.GetVolumeRange() #Needed to get the range of the volume (min, max) in dB (in this case: (-96.0, 0.0, 0.125) = (min = -96.0, max = 0.0)) #print(volume.GetVolumeRange()) to get the values
#volume.SetMasterVolumeLevel(-20.0, None) #Set the volume to -20.0 dB (it will set the volume to 27 in my HP-laptop), None is the time to change the volume (None = instant).
'''
volRange = volume.GetVolumeRange()
volume.SetMasterVolumeLevel(0.0, None)
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400 #For the volume bar (400 is the max value of the volume bar) (the pixel location)
volPer = 0 #For the volume bar (the percentage of the volume bar)

while True:
    success, img = cap.read()

    #Find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False) #draw=False to avoid drawing the circles on the chosen landmarks, because it is done below
    if len(lmList) != 0:
        #print(lmList[4], lmList[8]) #Print the position of the tip of the index finger and the tip of the thumb
        x1, y1 = lmList[4][1], lmList[4][2] #x1, y1 are the coordinates of the tip of the thumb finger
        x2, y2 = lmList[8][1], lmList[8][2] #x1, y1 are the coordinates of the tip of the index finger
        cx, cy = (x1+x2)//2, (y1+y2)//2 #cx, cy are the coordinates of the center of the line between the tip of the thumb and the tip of the index finger (center of the line between the two fingers)

        #Draw a circle on the tip of the thumb and the tip of the index finger
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED) #Draw a circle on the tip of the thumb
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED) #Draw a circle on the tip of the index finger
        #Draw a line between the tip of the thumb and the tip of the index finger
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        #Draw a circle on the center of the line between the tip of the thumb and the tip of the index finger
        cv2.circle(img, (cx, cy), 7, (0, 0, 255), cv2.FILLED)

        #Calculate the length of the line between the tip of the thumb and the tip of the index finger
        length = np.hypot(x2-x1, y2-y1)
        print("Length: ", int(length))

        #Gesture detection and volume control
        if length < 50:
            cv2.circle(img, (cx, cy), 7, (0, 255, 0), cv2.FILLED)

            #Hand range 50 - 250
            #Volume range -96 - 0
        vol = np.interp(length, [50, 250], [minVol, maxVol]) #Interpolate the volume between the min and max volume
        volume.SetMasterVolumeLevel(vol, None) #Set the volume to the interpolated value
        print("dB: ", vol)

        #for the volume bar
        volBar = np.interp(length, [50, 250], [400, 150]) #Interpolate the volume between the min and max volume
        volPer = np.interp(length, [50, 250], [0, 100]) #Interpolate the volume between the min and max volume for the percentage

    #Drawing a volume bar
    #cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3) #Draw a rectangle on the left side of the screen (starting point (50, 150), ending point (85, 400), color (0, 255, 0), thickness 3
    #cv2.rectangle(img, (50, int(vol)), (85, 400), (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 250, 0), 2)



    #Frame rate calculation
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    #Draw fps on image
    cv2.putText(img, f'fps: {str(int(fps))}', (10, 35), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)


    cv2.imshow("Image", img)
    #waitKey with q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



