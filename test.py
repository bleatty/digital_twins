from OpenGL.GL import *
from OpenGL.GLUT import *


def drawFun():
    glClear(GL_COLOR_BUFFER_BIT)
    glutWireTeapot(0.5)
    glFlush()


glutInit()
glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA)
glutInitWindowSize(400, 400)
glutCreateWindow(b"First")
glutDisplayFunc(drawFun)
glutMainLoop()

