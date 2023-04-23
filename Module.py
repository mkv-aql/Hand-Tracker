__author__ = 'mkv-aql'
import cv2
import mediapipe as mp
import time
import numpy as np

class Module():
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

    #Methods / Definitions

