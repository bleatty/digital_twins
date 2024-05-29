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
    model_data_visual_over_signal = pyqtSignal()

    def __init__(self):
        super(MyWindow, self).__init__()
        # 从文件中加载UI定义
        self.ui = uic.loadUi('digital_twins.ui')
        # 设置默认页面
        self.ui.setWindowTitle("数字孪生软件")
        #self.ui.tabWidget.setCurrentIndex(0)
        self.ui.showMaximized()
#         self.ui.label_20.setVisible(False)
#         self.ui.label_21.setVisible(False)
#         self.ui.label_22.setVisible(False)
#         self.ui.label_23.setVisible(False)
#         self.ui.label_24.setVisible(False)
#         self.ui.label_25.setVisible(False)
#         self.ui.label_26.setVisible(False)
#         self.ui.label_27.setVisible(False)
#
#         # 信号和槽的连接
#         # 数据预处理
#         self.ui.page_loaddata_pushButton_pre.clicked.connect(self.raw_data_pre)
#         self.ui.page_pre_pushButton_show.clicked.connect(self.pre_data_show)
#
#         # 数据增强
#         self.ui.page_gan_pushButton_start.clicked.connect(self.gan_model)
#         self.model_train_over_signal.connect(self.gan_model)
#         self.ui.page_gan_pushButton_show.clicked.connect(self.gan_show)
#
#         # 训练集生成
#         self.ui.trainset_pushButton.clicked.connect(self.trainset_construct)
#         # 模型保存
#         # self.ui.page_loaddata_pushButton_model_save.clicked.connect(self.save_model_msg)
#
#         # 训练分类模型
#         self.ui.pushButton_classify.clicked.connect(self.train_classify)
#         #
#         # # 退出
#         self.ui.pushButton_quit.clicked.connect(self.quit_app)
#
#     def raw_data_pre(self):
#         # 调用cwt函数，绘制数据的小波时频图'./SHIYANdata/train/{}/'.format(m-1)
#         cwt('D:/105program/physics-information/data', 12000, 1024)
#         QMessageBox.information(self, "原始数据小波变换", "原始数据小波变换完成！")
#
#     def pre_data_show(self):
#         if self.ui.radioButton.isChecked():
#             self.ui.label_21.setVisible(True)
#             self.ui.label_23.setVisible(True)
#             self.ui.label_25.setVisible(True)
#             self.ui.label_27.setVisible(True)
#             self.ui.label_1_1.setScaledContents(True)
#             self.ui.label_1_2.setScaledContents(True)
#             self.ui.label_1_3.setScaledContents(True)
#             self.ui.label_1_4.setScaledContents(True)
#             png1 = QtGui.QPixmap(r'.\SHIYANdata\train\0\0\0_1_7.png')
#             self.ui.label_1_1.setPixmap(png1)
#             self.ui.label_1_1.setScaledContents(True)
#             png2 = QtGui.QPixmap(r'.\SHIYANdata\train\1\1\1_1_7.png')
#             self.ui.label_1_2.setPixmap(png2)
#             self.ui.label_1_2.setScaledContents(True)
#             png3 = QtGui.QPixmap(r'.\SHIYANdata\train\2\2\2_1_7.png')
#             self.ui.label_1_3.setPixmap(png3)
#             self.ui.label_1_3.setScaledContents(True)
#             png4 = QtGui.QPixmap(r'.\SHIYANdata\train\3\3\3_1_7.png')
#             self.ui.label_1_4.setPixmap(png4)
#             self.ui.label_1_4.setScaledContents(True)
#         else:
#             self.ui.label_21.setVisible(True)
#             self.ui.label_23.setVisible(True)
#             self.ui.label_25.setVisible(True)
#             self.ui.label_27.setVisible(True)
#             self.ui.label_1_1.setScaledContents(True)
#             self.ui.label_1_2.setScaledContents(True)
#             self.ui.label_1_3.setScaledContents(True)
#             self.ui.label_1_4.setScaledContents(True)
#             png1 = QtGui.QPixmap(r'.\SHIYANdata\train\0\0\0_1_7.png')
#             self.ui.label_1_1.setPixmap(png1)
#             self.ui.label_1_1.setScaledContents(True)
#             png2 = QtGui.QPixmap(r'.\SHIYANdata\train\1\1\1_1_7.png')
#             self.ui.label_1_2.setPixmap(png2)
#             self.ui.label_1_2.setScaledContents(True)
#             png3 = QtGui.QPixmap(r'.\SHIYANdata\train\2\2\2_1_7.png')
#             self.ui.label_1_3.setPixmap(png3)
#             self.ui.label_1_3.setScaledContents(True)
#             png4 = QtGui.QPixmap(r'.\SHIYANdata\train\3\3\3_1_7.png')
#             self.ui.label_1_4.setPixmap(png4)
#             self.ui.label_1_4.setScaledContents(True)
#
#
#     def gan_model(self):
#         sub_thread = Thread(target=self.gan_train)
#         sub_thread.start()
#
#     def gan_show(self):
#         self.ui.label_20.setVisible(True)
#         self.ui.label_22.setVisible(True)
#         self.ui.label_24.setVisible(True)
#         self.ui.label_26.setVisible(True)
#         self.ui.label_2_1.setScaledContents(True)
#         self.ui.label_2_2.setScaledContents(True)
#         self.ui.label_2_3.setScaledContents(True)
#         self.ui.label_2_4.setScaledContents(True)
#         png = QtGui.QPixmap(r'.\Generate\0\0_7_7.png')  # 需要改地址
#         self.ui.label_2_1.setPixmap(png)
#         png = QtGui.QPixmap(r'.\Generate\1\1_7_7.png')
#         self.ui.label_2_2.setPixmap(png)
#         png = QtGui.QPixmap(r'.\Generate\2\2_7_7.png')
#         self.ui.label_2_3.setPixmap(png)
#         png = QtGui.QPixmap(r'.\Generate\3\3_7_7.png')
#         self.ui.label_2_4.setPixmap(png)
#
#     def trainset_construct(self):
#         # for i in range(4):
#         #     source_path = os.path.abspath(r'.\Generate\{}'.format(i))
#         #     target_path = os.path.abspath(r'.\SHIYANdata\train\{}\{}'.format(i, i))
#         #     if os.path.exists(source_path):
#         #         # root 所指的是当前正在遍历的这个文件夹的本身的地址
#         #         # dirs 是一个 list，内容是该文件夹中所有的目录的名字(不包括子目录)
#         #         # files 同样是 list, 内容是该文件夹中所有的文件(不包括子目录)
#         #         for root, dirs, files in os.walk(source_path):
#         #             for file in files:
#         #                 src_file = os.path.join(root, file)
#         #                 shutil.copy(src_file, target_path)
#         if os.path.exists(r'.\image\train'):
#             shutil.rmtree(r'.\image\train')  # 删除再建立
#             os.makedirs(r'.\image\train')
#         else:
#             os.makedirs(r'.\image\train')
#         for i in range(4):
#             source_path = os.path.abspath(r'.\Generate\{}'.format(i))
#             target_path = os.path.abspath(r'.\image\train')
#             if os.path.exists(source_path):
#                 # root 所指的是当前正在遍历的这个文件夹的本身的地址
#                 # dirs 是一个 list，内容是该文件夹中所有的目录的名字(不包括子目录)
#                 # files 同样是 list, 内容是该文件夹中所有的文件(不包括子目录)
#                 for root, dirs, files in os.walk(source_path):
#                     for file in files:
#                         src_file = os.path.join(root, file)
#                         shutil.copy(src_file, target_path)
#         for i in range(4):
#             source_path = os.path.abspath(r'.\SHIYANdata\train\{}\{}'.format(i, i))
#             target_path = os.path.abspath(r'.\image\train')
#             if os.path.exists(source_path):
#                 # root 所指的是当前正在遍历的这个文件夹的本身的地址
#                 # dirs 是一个 list，内容是该文件夹中所有的目录的名字(不包括子目录)
#                 # files 同样是 list, 内容是该文件夹中所有的文件(不包括子目录)
#                 for root, dirs, files in os.walk(source_path):
#                     for file in files:
#                         src_file = os.path.join(root, file)
#                         shutil.copy(src_file, target_path)
#         if os.path.exists(r'.\image\test'):
#             shutil.rmtree(r'.\image\test')  # 删除再建立
#             os.makedirs(r'.\image\test')
#         else:
#             os.makedirs(r'.\image\test')
#         for i in range(4):
#             source_path = os.path.abspath(r'.\SHIYANdata\test\{}\{}'.format(i, i))
#             target_path = os.path.abspath(r'.\image\test')
#             if os.path.exists(source_path):
#                 # root 所指的是当前正在遍历的这个文件夹的本身的地址
#                 # dirs 是一个 list，内容是该文件夹中所有的目录的名字(不包括子目录)
#                 # files 同样是 list, 内容是该文件夹中所有的文件(不包括子目录)
#                 for root, dirs, files in os.walk(source_path):
#                     for file in files:
#                         src_file = os.path.join(root, file)
#                         shutil.copy(src_file, target_path)
#         QMessageBox.information(self, "构造数据增强后的数据集", "数据集构造完成！")
#
#     def gan_train(self, nepoch=10001):
#         batch_size = 6
#         # imageSize = 28
#         nz = 1000
#         print("Random Seed: 88")
#         random.seed(88)
#         torch.manual_seed(88)
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         # 可以优化运行效率
#         cudnn.benchmark = True
#         if not os.path.exists('./img_VAE-GAN-SHIYAN'):
#             os.mkdir('./img_VAE-GAN-SHIYAN')
#         if not os.path.exists('./img_CWRU_true-SHIYAN'):
#             os.mkdir('./img_CWRU_true-SHIYAN')
#         for k in range(4):
#             # k代表图片为第几类
#             #####################################################
#             path_data_train = './SHIYANdata/train/{}'.format(k)
#             if not os.path.exists('./img_VAE-GAN-SHIYAN/{}'.format(k)):
#                 os.mkdir('./img_VAE-GAN-SHIYAN/{}'.format(k))
#             if not os.path.exists('./img_CWRU_true-SHIYAN/{}'.format(k)):
#                 os.mkdir('./img_CWRU_true-SHIYAN/{}'.format(k))
#             trainset = get_pic(path_data_train)  # 输入数据
#             dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
#             #####################################################
#             # print("=====> 构建VAE")
#             vae = VAE().to(device)
#             # vae.load_state_dict(torch.load('./CVAE-GAN-VAE.pth'))
#             # print("=====> 构建Discriminator")
#             D = Discriminator(1).to(device)
#             # D.load_state_dict(torch.load('./CVAE-GAN-Discriminator.pth'))
#             # 定义损失函数
#             criterion = nn.BCELoss().to(device)
#             # criterion = nn.BCELoss(reduction="sum").to(device)
#             criterionWithLogits = nn.BCEWithLogitsLoss().to(device)
#             CrossCriterion = nn.CrossEntropyLoss().to(device)  # 改用交叉熵损失
#             MSECriterion = nn.MSELoss().to(device)
#             # MSECriterion = nn.MSELoss(reduction="sum").to(device)
#             # print("=====> Setup optimizer")
#             # optimizerD = optim.Adam(D.parameters(), lr=0.0001)
#             optimizerD = optim.Adam(D.parameters(), lr=0.0001)
#             # optimizerVAE = optim.Adam(vae.parameters(), lr=0.0001)
#             optimizerVAE = optim.Adam(vae.parameters(), lr=0.0001)
#             k1 = 1
#             for epoch in range(nepoch):
#                 for i, (data, label) in enumerate(dataloader, 0):
#                     # print("epoch:",epoch)
#                     # 先处理一下数据
#                     data = data.to(device)  # data:torch.Size([batch_size, 3, 64, 64])  label:128
#                     # data = data[:,0,:,:]   # 只使用通道1的数据
#                     # data = data.unsqueeze(1) # 扩充维度
#                     # batch_size = data.shape[0]
#                     if k1:
#                         k1 = 0
#                         z, mean, logstd = vae.encoder(data)
#                         z = torch.randn(batch_size, nz).to(device)
#                         recon_data = vae.decoder(z)
#                         var = torch.pow(torch.exp(logstd), 2)
#                         MSE = MSECriterion(recon_data, data)
#                         BCE = criterionWithLogits(recon_data, data)
#                         print('开始******************************************')
#                         print("开始训练第{}类图片生成器".format(k))
#                         print('训练前 MSE: %.4f ' % (MSE.item()))
#                         print('训练前 BCE: %.4f ' % (BCE.item()))
#                         print('mean: %.4f' % (torch.mean(mean).item()))
#                         print('var: %.4f' % (torch.mean(var).item()))
#                         # print('image:')
#                         # print(data.shape)
#                         # print('fake_image:')
#                         # print(recon_data.shape)
#                         print('******************************************')
#                     # ***********************************************
#                     # 训练D
#                     # 直接使用D判别data的类型，使其与real_label接近
#                     output = D(data)  # 为什么这里输出的是[1,1,1,1]
#                     real_label = torch.ones(batch_size).to(device)  # 定义真实的图片label为1
#                     fake_label = torch.zeros(batch_size).to(device)  # 定义假的图片的label为0
#                     errD_real = criterion(output, real_label)
#                     # torch.randn:用来生成随机数字的tensor，这些随机数字满足标准正态分布（0~1）。
#                     # 使噪声接近错误标签
#                     z = torch.randn(batch_size, nz).to(device)  # 相当于随机噪声
#                     fake_data = vae.decoder(z)  # torch.Size([4, 3, 64, 64])
#                     output = D(fake_data)  # 4
#                     errD_fake = criterion(output, fake_label)
#                     errD = errD_real + errD_fake
#                     D.zero_grad()
#                     errD.backward()
#                     optimizerD.step()
#                     # ***********************************************
#                     # 更新VAE(G)1
#                     # 使重采样的数据接近data，标签更接近真实标签
#                     z, mean, logstd = vae.encoder(
#                         data)  # torch.Size([4, 100])  torch.Size([4, 100])  torch.Size([4, 100])
#                     recon_data = vae.decoder(z)
#                     vae_loss1, MSE, BCE = loss_function(recon_data, data, mean, logstd)
#                     # 更新VAE(G)2
#                     output = D(recon_data)  # tensor([1.7076e-28, 0.0000e+00, 2.0358e-32, 4.8602e-19]
#                     # output = D(recon_data).detach()
#                     real_label = torch.ones(batch_size).to(device)  # [1,1,1,1]
#                     vae_loss2 = criterion(output, real_label)
#                     # vae_loss2 = criterionWithLogits(output, real_label)  # 加入softmax
#                     vae.zero_grad()
#                     vae_loss = vae_loss1 + vae_loss2
#                     # vae_loss = vae_loss/100  # 进行尺度缩放
#                     vae_loss.backward()
#                     optimizerVAE.step()
#                     if epoch == 0:
#                         real_images = make_grid(data.cpu(), nrow=8, normalize=True).detach()
#                         for i in range(batch_size):
#                             data_one = data[i]
#                             true_images = make_grid(data_one.cpu(), nrow=8, normalize=True).detach()
#                             # 获取今天的字符串
#                             now_time = int(time.time())
#                             save_image(true_images, r'./img_CWRU_true-SHIYAN/{}/{}.png'.format(k, now_time))
#                         # save_image(real_images, "./img_CVAE-GAN/real_images.png")
#                     if i == len(dataloader) - 1:
#                         sample = torch.randn(data.shape[0], nz).to(device)
#                         output = vae.decoder(sample)
#                         for i in range(batch_size):
#                             output_one = output[i]
#                             fake_images = make_grid(output_one.cpu(), nrow=8, normalize=True).detach()
#                             # 获取今天的字符串
#                             now_time = int(time.time())  # 返回从1970纪元到当前时刻经过秒的整数
#                             save_image(fake_images, r'./img_VAE-GAN-SHIYAN/{}/{}.png'.format(k, now_time))
#                         # fake_images = make_grid(output.cpu(), nrow=8, normalize=True).detach()
#                         # save_image(fake_images, './img_CVAE-GAN/{}/{}.png'.format(label_one,epoch))
#                 # print("vae_loss1:%f,vae_loss2:%f,vae_loss3:%f" % (vae_loss1, vae_loss2, vae_loss3))
#                 if epoch % 50 == 0:
#                     var = torch.pow(torch.exp(logstd), 2)
#                     print('[%d/%d] Loss_D: %.4f Loss_G: %.4f MSE: %.4f BCE: %.4f '
#                           % (epoch, nepoch - 1, errD.item(), vae_loss.item(), MSE.item(), BCE.item()))
#                     print('mean: %.4f' % (torch.mean(mean).item()))
#                     print('var: %.4f' % (torch.mean(var).item()))
#             print("开始生成第{}类图片".format(k))
#             sample = torch.randn(700, nz).to(device)
#             output = vae.decoder(sample)
#             if not os.path.exists('./Generate'):
#                 os.mkdir('./Generate')
#             # 为每个类别创建文件夹，保存生成的图像
#             if not os.path.exists('./Generate/{}'.format(k)):
#                 os.mkdir('./Generate/{}'.format(k))
#             for i in range(700):
#                 output_one = output[i]
#                 fake_images = make_grid(output_one.cpu(), nrow=8, normalize=True).detach()
#                 save_image(fake_images, './Generate/{}/{}_{}_7.png'.format(k, k, i + 6))
#             torch.save(vae.state_dict(), './VAE-GAN-VAE-{}.pth'.format(k))
#             torch.save(D.state_dict(), './VAE-GAN-Discriminator-{}.pth'.format(k))
#             print('结束******************************************')
#         self.model_train_over_signal.emit()
#         QMessageBox.information(self, "数据增强", "数据增强完成！")
#
#     def train_classify(self):
#         train_x, train_y = readfile("./image/train")
#         test_x, test_y = readfile("./image/test")
#         # training 时做 data augmentation
#         # transforms.Compose 将图像操作串联起来
#         train_transform = transforms.Compose([
#             transforms.ToPILImage(),  # 限制在（0，1）之间
#             transforms.RandomHorizontalFlip(),  # 随机将图片水平翻转
#             transforms.RandomRotation(15),  # 随机旋转图片 (-15,15)
#             transforms.ToTensor(),  # 将图片转成 Tensor, 并把数值normalize到[0,1](data normalization)
#         ])
#         # testing 时不需做 data augmentation
#         test_transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.ToTensor(),
#         ])
#
#         batch_size = 10
#         train_set = ImgDataset(train_x, train_y, train_transform)
#         test_set = ImgDataset(test_x, test_y, test_transform)
#         train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
#         test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
#
#         # Alex
#         net1 = nn.Sequential(
#             # 这里，我们使用一个11*11的更大窗口来捕捉对象。
#             # 同时，步幅为4，以减少输出的高度和宽度。
#             # 另外，输出通道的数目远大于LeNet
#             nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
#             nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             # 使用三个连续的卷积层和较小的卷积窗口。
#             # 除了最后的卷积层，输出通道的数量进一步增加。
#             # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
#             nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
#             nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Flatten(),
#             # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
#             nn.Linear(6400, 4096), nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(4096, 4096), nn.ReLU(),
#             nn.Dropout(p=0.5),
#             # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
#             nn.Linear(4096, 4))
#         print(net1)
#
#         # 构造一个单通道数据，来观察每一层输出的形状
#
#         # x = torch.randn(1, 3, 224, 224)
#         # for layer in net1:
#         #     x = layer(x)
#         #     print(layer.__class__.__name__, 'output shape:\t', x.shape)
#
#         lr, num_epochs = 0.01, 30
#         d2l.train_ch6(net1, train_loader, test_loader, num_epochs, lr, d2l.try_gpu())
#         d2l.plt.savefig('alexnet.png',dpi=600)
#         #d2l.plt.show()
#         self.ui.label.setScaledContents(True)
#         png = QtGui.QPixmap('alexnet.png')  # 需要改地址
#         self.ui.label.setPixmap(png)
#         QMessageBox.information(self, "故障诊断", "故障诊断完成！")
#
#
#     def quit_app(self):
#         QApplication.quit()
#
#
# # 获得小波频谱图
# def cwt(path, fs, N):
#     dirs = os.listdir(path)
#     m = 0
#     title = ['X118_DE_time', 'X105_DE_time', 'X097_DE_time', 'X130_DE_time']
#     for i in dirs:
#         fullname = os.path.join(path, i)
#         A = scipy.io.loadmat(fullname)
#         data_raw = A[title[m]]
#         data = (data_raw - np.min(data_raw)) / (np.max(data_raw) - np.min(data_raw))
#         m += 1
#         k = 0
#         # 构造训练集文件夹
#         if os.path.exists('./SHIYANdata/train/{}'.format(m - 1)):
#             shutil.rmtree('./SHIYANdata/train/{}'.format(m - 1))  # 删除再建立
#             os.makedirs('./SHIYANdata/train/{}'.format(m - 1))
#         else:
#             os.makedirs('./SHIYANdata/train/{}'.format(m - 1))
#         if os.path.exists('./SHIYANdata/train/{}/{}'.format(m - 1, m - 1)):
#             shutil.rmtree('./SHIYANdata/train/{}/{}'.format(m - 1, m - 1))  # 删除再建立
#             os.makedirs('./SHIYANdata/train/{}/{}'.format(m - 1, m - 1))
#         else:
#             os.makedirs('./SHIYANdata/train/{}/{}'.format(m - 1, m - 1))
#         # 构造测试集文件夹
#         if os.path.exists('./SHIYANdata/test/{}'.format(m - 1)):
#             shutil.rmtree('./SHIYANdata/test/{}'.format(m - 1))  # 删除再建立
#             os.makedirs('./SHIYANdata/test/{}'.format(m - 1))
#         else:
#             os.makedirs('./SHIYANdata/test/{}'.format(m - 1))
#         if os.path.exists('./SHIYANdata/test/{}/{}'.format(m - 1, m - 1)):
#             shutil.rmtree('./SHIYANdata/test/{}/{}'.format(m - 1, m - 1))  # 删除再建立
#             os.makedirs('./SHIYANdata/test/{}/{}'.format(m - 1, m - 1))
#         else:
#             os.makedirs('./SHIYANdata/test/{}/{}'.format(m - 1, m - 1))
#         for j in range(5120, 8192, 512):
#             data1 = data[j:(j + 1023)]
#             data1 = data1.flatten()
#             k = k + 1
#             wavename = 'cmor3-3'
#             fc = pywt.central_frequency(wavename)
#             totalscal = 256
#             cparam = 2 * fc * totalscal
#             scales = cparam / np.arange(totalscal, 1, -1)
#             [cwtmatr, frequencies] = pywt.cwt(data1, scales, wavename, 1.0 / fs)  # 求连续小波系数
#             t = np.linspace(0, N - 1 / fs, N - 1)
#             plt.contourf(t, frequencies, abs(cwtmatr))
#             plt.gcf().set_size_inches(1023 / 100, 1023 / 100)
#             plt.gca().xaxis.set_major_locator(plt.NullLocator())
#             plt.gca().yaxis.set_major_locator(plt.NullLocator())
#             plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
#             plt.margins(0, 0)
#             plt.savefig(
#                 './SHIYANdata/train/{}/{}/'.format(m - 1, m - 1) + str(m - 1) + '_' + str(k - 1) + '_7' + ".png")
#             plt.clf()
#         k = 0
#         for j in range(40960, 86528, 512):
#             data1 = data[j:(j + 1023)]
#             data1 = data1.flatten()
#             k = k + 1
#             wavename = 'cmor3-3'
#             fc = pywt.central_frequency(wavename)
#             totalscal = 256
#             cparam = 2 * fc * totalscal
#             scales = cparam / np.arange(totalscal, 1, -1)
#             [cwtmatr, frequencies] = pywt.cwt(data1, scales, wavename, 1.0 / fs)  # 求连续小波系数
#             t = np.linspace(0, N - 1 / fs, N - 1)
#             plt.contourf(t, frequencies, abs(cwtmatr))
#             plt.gcf().set_size_inches(1023 / 100, 1023 / 100)
#             plt.gca().xaxis.set_major_locator(plt.NullLocator())
#             plt.gca().yaxis.set_major_locator(plt.NullLocator())
#             plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
#             plt.margins(0, 0)
#             plt.savefig('./SHIYANdata/test/{}/{}/'.format(m - 1, m - 1) + str(m - 1) + '_' + str(k - 1) + '_7' + ".png")
#             plt.clf()
#
#
# # 生成器构造
# class VAE(nn.Module):
#     def __init__(self):
#         super(VAE, self).__init__()
#         # 定义编码器
#         self.encoder_conv = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(16),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#         self.encoder_fc1 = nn.Linear(64 * img_size_8 * img_size_8, nz)
#         self.encoder_fc2 = nn.Linear(64 * img_size_8 * img_size_8, nz)
#         self.Sigmoid = nn.Sigmoid()
#         # self.decoder_fc = nn.Linear(nz+num_class,32 * 7 * 7)
#         self.decoder_fc = nn.Linear(nz, 64 * img_size_8 * img_size_8)
#         self.decoder_deconv = nn.Sequential(
#             nn.ConvTranspose2d(64, 32, 4, 2, 1),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(32, 16, 4, 2, 1),
#             nn.ReLU(inplace=True),
#             # nn.ConvTranspose2d(16, 1, 4, 2, 1),
#             nn.ConvTranspose2d(16, 3, 4, 2, 1),
#             nn.Sigmoid(),
#         )
#
#     def noise_reparameterize(self, mean, logvar):
#         # 生成噪音信号
#         # eps = torch.randn(mean.shape).to(device)
#         # z = mean + eps * torch.exp(logvar)
#         # return z
#         eps = torch.randn(mean.size(0), mean.size(1)).cuda()
#         z = mean + eps * torch.exp(logvar / 2)
#         return z
#
#     def forward(self, x):
#         z = self.encoder(x)
#         output = self.decoder(z)
#         return output
#
#     def encoder(self, x):
#         # print("x:",x)
#         out1, out2 = self.encoder_conv(x), self.encoder_conv(x)
#         # torch.Size([10, 3, 64, 64]) -> torch.Size([10, 32, 16, 16])
#         # print("out1:",out1)
#         # print("out2:",out2)
#         mean = self.encoder_fc1(out1.view(out1.shape[0], -1))
#         logstd = self.encoder_fc2(out2.view(out2.shape[0], -1))
#         z = self.noise_reparameterize(mean, logstd)
#         return z, mean, logstd
#
#     def decoder(self, z):
#         # torch.Size([10, 111])
#         out3 = self.decoder_fc(z)  # torch.Size([10, 8192])
#         # out3 = out3.view(out3.shape[0], 32, 7, 7)
#         out3 = out3.view(out3.shape[0], 64, img_size_8, img_size_8)  # torch.Size([10, 32, 16, 16])
#         out3 = self.decoder_deconv(out3)  # torch.Size([10, 3, 64, 64])
#         return out3
#
#
# # 判别器构造
# class Discriminator(nn.Module):
#     def __init__(self, outputn=1):
#         super(Discriminator, self).__init__()
#         self.dis = nn.Sequential(
#             # nn.Conv2d(1, 32, 3, stride=1, padding=1),
#             nn.Conv2d(3, 32, 3, stride=1, padding=1),
#             nn.LeakyReLU(0.2, True),
#             nn.MaxPool2d((2, 2)),
#
#             nn.Conv2d(32, 64, 3, stride=1, padding=1),
#             nn.LeakyReLU(0.2, True),
#             nn.MaxPool2d((2, 2)),
#         )
#         self.fc = nn.Sequential(
#             # nn.Linear(7 * 7 * 64, 1024),
#             nn.Linear(img_size_4 * img_size_4 * 64, 1024),
#             nn.LeakyReLU(0.2, True),
#             # nn.ReLU(inplace=True),
#             nn.Linear(1024, outputn),
#             nn.Sigmoid()
#         )
#
#     def forward(self, input):
#         x = self.dis(input)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x.squeeze(1)
#
#
# # 构造损失函数
# def loss_function(recon_x, x, mean, logstd):
#     # BCE = F.binary_cross_entropy(recon_x,x,reduction='sum')
#     # BCE = F.binary_cross_entropy(recon_x,x,reduction='mean')  # 默认reduction='mean'
#
#     BCE = criterionWithLogits(recon_x, x)
#     MSE = MSECriterion(recon_x, x)
#     # 因为var是标准差的自然对数，先求自然对数然后平方转换成方差
#     var = torch.pow(torch.exp(logstd), 2)
#     KLD = -0.5 * torch.sum(1 + torch.log(var) - torch.pow(mean, 2) - var)  # 原始版本
#     # KLD = -0.5 * torch.sum(1+torch.log(var)-torch.pow(mean,2)-torch.exp(logstd))  # keras更改
#     # KLD = 0.5 * torch.sum(torch.exp(var)+torch.pow(mean,2) - 1 - var)
#     # return MSE+KLD
#     loss = BCE + KLD
#     return loss, MSE, BCE
#
#
# # 构建dataloader
# def get_pic(path):
#     transform = transforms.Compose([transforms.Resize((img_size, img_size)),
#                                     transforms.ToTensor()
#                                     ])
#     # onvert a PIL image to tensor (H*W*C) in range [0,255] to a torch.Tensor(C*H*W) in the range [0.0,1.0]
#     dataset = torchvision.datasets.ImageFolder(path, transform=transform)
#     return (dataset)
#
#
# class ImgDataset(Dataset):
#     def __init__(self, x, y=None, transform=None):
#         self.x = x
#         # label is required to be a LongTensor
#         # torch.FloatTensor是32位浮点类型数据，torch.LongTensor是64位整型
#         self.y = torch.LongTensor(y)
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.x)
#
#     def __getitem__(self, index):
#         X = self.transform(self.x[index])
#         Y = self.y[index]
#         return X, Y
#
#
# # Read image 利用 OpenCV(cv2) 读入照片并存放在 numpy array 中
# def readfile(path):
#     # label 是一个 boolean variable, 代表需不需要回传 y 值
#     image_dir = sorted(os.listdir(path))  # os.listdir(path)将path路径下的文件名以列表形式读出
#     # print(os.listdir(path))
#     # print(image_dir)
#     x = np.zeros((len(image_dir), 224, 224, 3), dtype=np.uint8)
#     y = np.zeros((len(image_dir)), dtype=np.uint8)
#     for i, file in enumerate(image_dir):
#         img = cv2.imread(os.path.join(path, file))  # os.path.join(path, file) 路径名合并
#         x[i, :, :] = cv2.resize(img, (224, 224))
#         y[i] = int(file.split("_")[0])  # split("")[0] 得到的是第一个_之前的内容，split("")[1] 得到的是第一个_和第二个_之间的内容
#     return x, y
#
#
# # 训练网络





if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    myshow = MyWindow()
    extra = {
        # Font
        'font_family': 'monoespace',
        'font_size': '20px',
        'line_height': '20px',
        # Density Scale
        'density_scale': '0',
    }
    # setup stylesheet
    apply_stylesheet(app, theme='light_blue.xml',invert_secondary=True,extra=extra)
    myshow.ui.show()
    sys.exit(app.exec_())
