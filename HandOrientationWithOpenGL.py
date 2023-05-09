import cv2
import mediapipe as mp
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import OpenGLModule as om

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)


if not glfw.init():
    raise Exception("glfw can not be initialized!")

# Create the window
window = glfw.create_window(640, 480, "Hand Tracking with OpenGL Arrow", None, None)
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")

glfw.set_window_pos(window, 640, 480)
glfw.make_context_current(window)

def hand_landmarks_to_gl_coords(landmark, window_size=(1280, 720), z_scale=1.0):
    x_gl = (landmark.x - 0.5) * 2.0
    y_gl = (0.5 - landmark.y) * 2.0
    z_gl = (0.5 - landmark.z * z_scale) * 2.0
    return x_gl, y_gl, z_gl


with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while not glfw.window_should_close(window):
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # OpenGL rendering
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate the palm center using landmarks 0, 5, 9, 13, and 17
                palm_center = np.mean(
                    [hand_landmarks_to_gl_coords(hand_landmarks.landmark[i]) for i in [0, 5, 9, 13, 17]],
                    axis=0
            )

            # Draw the arrow
            om.drawArrow(scale = 0.15, centerCoords=palm_center, color=(1.0, 1.0, 0.0))

        # Swap buffers and poll events
        glfw.swap_buffers(window)
        glfw.poll_events()

        # Display the frame with hand landmarks in a separate window
        frame_with_landmarks = frame.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_with_landmarks, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow("Hand Tracking", frame_with_landmarks)

        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
glfw.terminate()