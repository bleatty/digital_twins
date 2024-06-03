from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

def draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glRotatef(2, 1, 1, 1)
    glutSolidTeapot(0.5)
    glFlush()


if __name__ == '__main__':
    glutInit()
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
    glutInitWindowSize(400, 400)
    glutCreateWindow(b"PyOpenGL Example")
    gluOrtho2D(-1, 1, -1, 1)
    glutDisplayFunc(draw)
    glutMainLoop()
