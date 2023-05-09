__author__ = 'mkv-aql'
import cv2
import mediapipe as mp
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

class OpenGLRenderer:
    def __init__(self, window_width, window_height):
        if not glfw.init():
            raise Exception("glfw cannot be initialized!")

        self.window = glfw.create_window(window_width, window_height, "Hand Tracking with OpenGL Arrow", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("glfw window cannot be created!")

        glfw.set_window_pos(self.window, 640, 480)
        glfw.make_context_current(self.window)
        glEnable(GL_DEPTH_TEST)

    def window_should_close(self):
        return glfw.window_should_close(self.window)

    def clear(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.0, 0.0, 0.0, 1.0)

    def swap_buffers(self):
        glfw.swap_buffers(self.window)

    def poll_events(self):
        glfw.poll_events()

    def terminate(self):
        glfw.terminate()

#Drawing Shapes
    def draw_arrow(self, scale, center_coords, color=(1.0, 0.0, 0.0)):
        glColor3f(color[0], color[1], color[2])
        glPushMatrix()
        glTranslatef(center_coords[0], center_coords[1], center_coords[2])
        glScalef(scale, scale, scale)

        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(1.0, 0.0, 0.0)
        glEnd()

        glBegin(GL_TRIANGLES)
        glVertex3f(1.0, 0.0, 0.0)
        glVertex3f(0.85, 0.1, 0.0)
        glVertex3f(0.85, -0.1, 0.0)
        glEnd()

        glPopMatrix()

    def drawArrow(self, scale, centerCoords, color=(1.0, 0.0, 0.0)):
        glColor3f(color[0], color[1], color[2])  # Set the color to red
        glPushMatrix()
        glTranslate(centerCoords[0], centerCoords[1], centerCoords[2])
        glScale(scale, scale, scale)

        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(1.0, 0.0, 0.0)
        glEnd()

        glBegin(GL_TRIANGLES)
        glVertex3f(1.0, 0.0, 0.0)
        glVertex3f(0.85, 0.1, 0.0)
        glVertex3f(0.85, -0.1, 0.0)
        glEnd()

        glPopMatrix()

cap = cv2.VideoCapture(0)

renderer = OpenGLRenderer(window_width=640, window_height=480)

def hand_landmarks_to_gl_coords(landmark, window_size=(1280, 720), z_scale=1.0):
    x_gl = (landmark.x - 0.5) * 2.0
    y_gl = (0.5 - landmark.y) * 2.0
    z_gl = (0.5 - landmark.z * z_scale) * 2.0
    return x_gl, y_gl, z_gl

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while not renderer.window_should_close():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        renderer.clear()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                palm_center = np.mean(
                    [hand_landmarks_to_gl_coords(hand_landmarks.landmark[i]) for i in [0, 5, 9, 13, 17]],
                    axis=0
            )
            # Draw the arrow
            renderer.draw_arrow(scale=0.15, center_coords=palm_center, color=(1.0, 1.0, 0.0))

        # Swap buffers and poll events
        renderer.swap_buffers()
        renderer.poll_events()

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
renderer.terminate()
