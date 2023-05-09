__author__ = 'mkv-aql'
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(False) # static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5), If static_image_mode=True, it will continously detect hands (slower performance)
mpDraw = mp.solutions.drawing_utils
#For fps
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #Extracting the landmarks from results
    #print(results.multi_hand_landmarks)
        #If there are hands detected, for each hand (handLms), draw the landmarks and show id no. with its location.
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm) #id is the landmark id no. and lm is the landmark location. id = 0 is the palm of the hand, id = 4 is thw tip of the thumb
                h, w, c = img.shape
                cx, cy, cz = int(lm.x*w), int(lm.y*h), lm.z
                print(id, cx, cy, cz) #id is the landmark id no. and lm is the landmark location. id = 0 is the palm of the hand, id = 4 is thw tip of the thumb
                if id == 4:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED) #Draw a circle on the tip of the thumb

    #For fps calculation
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    #Draw fps on image
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 2)




    cv2.imshow("Image", img)
    #waitKey with q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


