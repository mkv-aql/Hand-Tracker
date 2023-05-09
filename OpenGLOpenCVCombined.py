__author__ = 'mkv-aql'
import cv2
import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import OpenGLModule as om

width, height = 640, 480


def main():
    if not glfw.init():
        return

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(width, height, "Hidden window", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (width / height), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)
    glEnable(GL_DEPTH_TEST)

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    angle = 0.0

    while True:
        # Capture a frame from the webcam
        ret, webcam_frame = cap.read()
        if not ret:
            break

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glRotatef(angle, 3, 1, 1)

        om.drawCube(scale = 0.5, centerCoords = (0, 0, 0))

        glfw.swap_buffers(window)
        glfw.poll_events()

        # Read pixel data and convert it to an OpenCV image
        pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.flip(image, 0)

        # Combine the OpenGL-rendered cube and the webcam frame
        #image = cv2.addWeighted(image, 0.6, webcam_frame, 0.4, 0) # 60% of the cube, 40% of the webcam frame, no gamma correction
        image = cv2.addWeighted(image, 1, webcam_frame, 1, 1.5) # 100% of the cube, 40% of the webcam frame, gamma correction of 1.5

        cv2.imshow("OpenGL/OpenCV/Webcam Example", image)
        key = cv2.waitKey(1)

        if key == 27:  # Escape key to exit
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        angle += 0.5

    cap.release()
    glfw.terminate()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
