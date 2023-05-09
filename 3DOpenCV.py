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
    gluPerspective(45, (width / height), 0.1, 100.0)
    glTranslatef(0.0, 0.0, -1)
    glEnable(GL_DEPTH_TEST)

    angle = 0.0

    while True:
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glRotatef(angle, 0.1, 0.1, 0)

        om.drawCube(0.05, (0,0,0))

        glfw.swap_buffers(window)
        glfw.poll_events()

        # Read pixel data and convert it to an OpenCV image
        pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.flip(image, 0)

        cv2.imshow("OpenGL/OpenCV Example", image)
        key = cv2.waitKey(1)

        if key == 27:  # Escape key to exit
            break

        angle += 0.5

    glfw.terminate()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()