from PyQt5.QtWidgets import (QWidget, QComboBox, QAction, qApp, QMainWindow, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QSizePolicy, QGraphicsBlurEffect)
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt
from video_widget import VideoWidget
from right_label import RightLabel
from get_camera import Camera
from some_functions import resizeImage, toQImage

import scipy.misc
import time
import numpy as np


class MainWidget(QWidget):

    def __init__(self, video_widget, right_label, camera):
        super().__init__()
        self.setGeometry(100, 100, 1200, 600)

        self.camera = camera
        self.video_widget = video_widget
        self.right_label = right_label

        self.portrait = 0

        self.initUI()

    def initUI(self):
        image_r = resizeImage("images/desktop.jpg", 600)
        image_r = QPixmap.fromImage(image_r)
        self.right_label.setPixmap(image_r)

        h = QHBoxLayout()
        h.addWidget(self.video_widget)
        h.addWidget(self.right_label)
        self.setLayout(h)

    def activate(self):
        self.video_widget.activate()

    def deactivate(self):
        self.hide()
        self.video_widget.deactivate()
