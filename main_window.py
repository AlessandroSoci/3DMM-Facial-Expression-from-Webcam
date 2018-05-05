from PyQt5.QtWidgets import (QWidget, QComboBox, QAction, qApp, QMainWindow, QHBoxLayout, QVBoxLayout, QLabel, QSizePolicy, QSlider, QProgressBar)
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt
from get_camera import Camera
from video_widget import VideoWidget
from right_label import RightLabel
from some_functions import toQImage, resizeImage
from main_widget import MainWidget

from expression_code.main_expression import *

import cv2
import sys
sys.path.append('expression_code/')
import scipy.misc
import numpy as np


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = "3DMM"
        self.setWindowIcon(QtGui.QIcon('images/1UP.ico'))
        self.setFixedSize(1200, 700)

        self.camera = Camera()
        self.progress_bar = QProgressBar(self)
        self.count = 0

        self.toolbar = self.addToolBar('Main Window')
        # self.toolbar_emotions = self.addToolBar('Emotions')

        self.expression = "neutral"
        self.picture_taken = False
        self.bool_onclick = None
        self.portrait = None
        self.model = Model(None, self.expression, self)
        self.video_widget = VideoWidget(self.camera)
        self.right_label = RightLabel(self, self.model)
        self.main_widget = MainWidget(self.video_widget, self.right_label, self.camera)
        self.setCentralWidget(self.main_widget)

        self.model.progress_bar.connect(self.set_progress_bar, type=Qt.QueuedConnection)

        self.initUI()

    def initUI(self):
        self.build_toolbar()
        # self.addToolBarBreak()

    def get_expression(self):
        return self.expression

    def get_bool_onclick(self):
        return self.bool_onclick

    def set_progress_bar(self):
        if self.count <= 2:
            self.progress_bar.setValue(self.count * 50)
            self.count = self.count + 1
            return
        self.count = 0
        self.set_progress_bar()

    def build_toolbar(self):

        slide_bar = QSlider(Qt.Horizontal)
        slide_bar.setMinimum(0)
        slide_bar.setMaximum(8)
        slide_bar.setValue(0)
        slide_bar.valueChanged.connect(self.camera.setZoom)
        self.toolbar.addWidget(slide_bar)

        take_photo = QAction(QIcon('images/get_photo.png'), 'Take_Picture', self)
        take_photo.setShortcut('Ctrl+Q')
        take_photo.triggered.connect(self.on_click)
        self.toolbar.addAction(take_photo)

        # self.progress_bar.setHidden(True)
        self.toolbar.addWidget(self.progress_bar)

        self.toolbar.addSeparator()

        # build slide fo upload pre-built patterns.looking for the pattern class
        combo_box = QComboBox(self)
        combo_box.addItem("Neutral")
        combo_box.addItem("Surprise")
        combo_box.addItem("Happy")
        combo_box.addItem("Contempt")
        combo_box.addItem("Sadness")
        combo_box.addItem("Disgust")
        combo_box.addItem("Angry")
        combo_box.addItem("Fear")
        self.toolbar.addWidget(combo_box)
        combo_box.activated[str].connect(self.combo_changed)
        self.toolbar.addSeparator()

    def activate(self):
        self.main_widget.activate()
        self.show()

    def deactivate(self):
        self.main_widget.deactivate()

    def closeEvent(self, event):
        self.camera.deactivate()
        self.main_widget.deactivate()
        event.accept()

    def on_click(self):
        # self.progress_bar.setHidden(False)
        self.picture_taken = True
        self.bool_onclick = True
        self.portrait = self.camera.get_current_frame()
        self.portrait = resizeImage(self.portrait, 256)
        scipy.misc.imsave('expression_code/imgs/outfile.jpg', self.portrait)
        self.model.set_image('expression_code/imgs/outfile.jpg')
        self.right_label.activate()

    def combo_changed(self, text):
        self.bool_onclick = False
        self.expression = text.lower()
        if self.picture_taken:
            self.right_label.activate()
