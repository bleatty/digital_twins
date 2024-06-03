# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/25 17:00
# @Author  : lixinye
# @File    : main_new.py
import scipy.io
import pywt
import matplotlib.pyplot as plt
import random
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
from PyQt5.QtOpenGL import QGLWidget
from torchvision.utils import save_image
from torchvision.utils import make_grid
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import *
from PyQt5 import uic
from threading import Thread
from PyQt5.QtCore import pyqtSignal
from qt_material import apply_stylesheet
import pickle
import shutil
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import time
from d2l import torch as d2l
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

from PyQt5 import QtCore, QtGui, QtWidgets, QtOpenGL
from PyQt5.QtWidgets import QApplication, QMainWindow

# 网络设置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterionWithLogits = nn.BCEWithLogitsLoss().to(device)
CrossCriterion = nn.CrossEntropyLoss().to(device)  # 改用交叉熵损失
MSECriterion = nn.MSELoss().to(device)


# static function
# 保存
def save_dict(fname, dict_data):
    with open(fname, "wb") as f:
        pickle.dump(dict_data, f)


# 读取
def read_dict(fname):
    with open(fname, "rb") as f:
        d = pickle.load(f)
    return d


class MyWindow(QWidget):
    model_train_over_signal = pyqtSignal()  # 多线程协作信号，为了程序稳定我们一般不在子线程更改界面ui
    # model_data_visual_over_signal = pyqtSignal()

    def __init__(self):
        super(MyWindow, self).__init__()
        # 从文件中加载UI定义
        self.ui = uic.loadUi('digital_twins.ui')
        # 设置默认页面
        self.ui.setWindowTitle("数字孪生软件")
        self.ui.setGeometry(100, 100, 800, 600)  # 设置窗口的位置和大小
        #self.ui.tabWidget.setCurrentIndex(0)  # 如果有两面，将页面设置默认显示第一面
        self.ui.showMaximized()


        # 信号和槽的连接
        ## 连接opengl

        # 有限元替代模型
        self.ui.pushButton_ansys.clicked.connect(self.ansys_model)
        # self.model_train_over_signal.connect(self.model_show()) # 备用，用于动态展示模型训练结果

        # 左右移动
        self.ui.pushButton_left.clicked.connect(self.left_move)
        self.ui.pushButton_right.clicked.connect(self.left_move)

        # # 退出
        self.ui.pushButton_quit.clicked.connect(self.quit_app)


#
    def quit_app(self):
        QApplication.quit()


    # 定义替代模型
    def ansys_model(self):
        return


    def model_show(self):
        return

    def left_move(self):
        return


    def right_move(self):
        return

# # 训练网络


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    myshow = MyWindow()
    extra = {
        # Font
        'font_family': 'monoespace',
        'font_size': '40px',
        'line_height': '40px',
        # Density Scale
        'density_scale': '0',
    }
    # setup stylesheet
    apply_stylesheet(app, theme='light_blue.xml',invert_secondary=True,extra=extra)
    myshow.ui.show()
    sys.exit(app.exec_())
