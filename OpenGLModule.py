__author__ = 'mkv-aql'
import cv2
import mediapipe as mp
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np

def drawCube(scale, centerCoords):
    scale = scale
    x = centerCoords[0]
    y = centerCoords[1]
    z = centerCoords[2]

    glBegin(GL_QUADS)

    glColor3f(0.0, 1.0, 0.0) #Green
    glVertex3f(x+1.0*scale, y+1.0*scale, z-1.0*scale)
    glVertex3f(x-1.0*scale, y+1.0*scale, z-1.0*scale)
    glVertex3f(x-1.0*scale, y+1.0*scale, z+1.0*scale)
    glVertex3f(x+1.0*scale, y+1.0*scale, z+1.0*scale)

    glColor3f(1.0, 0.5, 0.0) #
    glVertex3f(x+1.0*scale, y-1.0*scale, z+1.0*scale)
    glVertex3f(x-1.0*scale, y-1.0*scale, z+1.0*scale)
    glVertex3f(x-1.0*scale, y-1.0*scale, z-1.0*scale)
    glVertex3f(x+1.0*scale, y-1.0*scale, z-1.0*scale)

    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(x+1.0*scale, y+1.0*scale, z+1.0*scale)
    glVertex3f(x-1.0*scale, y+1.0*scale, z+1.0*scale)
    glVertex3f(x-1.0*scale, y-1.0*scale, z+1.0*scale)
    glVertex3f(x+1.0*scale, y-1.0*scale, z+1.0*scale)

    glColor3f(1.0, 1.0, 0.0)
    glVertex3f(x+1.0*scale, y-1.0*scale, z-1.0*scale)
    glVertex3f(x-1.0*scale, y-1.0*scale, z-1.0*scale)
    glVertex3f(x-1.0*scale, y+1.0*scale, z-1.0*scale)
    glVertex3f(x+1.0*scale, y+1.0*scale, z-1.0*scale)

    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(x-1.0*scale, y+1.0*scale, z+1.0*scale)
    glVertex3f(x-1.0*scale, y+1.0*scale, z-1.0*scale)
    glVertex3f(x-1.0*scale, y-1.0*scale, z-1.0*scale)
    glVertex3f(x-1.0*scale, y-1.0*scale, z+1.0*scale)

    glColor3f(1.0, 0.0, 1.0)
    glVertex3f(x+1.0*scale, y+1.0*scale, z-1.0*scale)
    glVertex3f(x+1.0*scale, y+1.0*scale, z+1.0*scale)
    glVertex3f(x+1.0*scale, y-1.0*scale, z+1.0*scale)
    glVertex3f(x+1.0*scale, y-1.0*scale, z-1.0*scale)

    glEnd()
