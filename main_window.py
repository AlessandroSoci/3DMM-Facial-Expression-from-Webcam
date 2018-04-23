from PyQt5.QtWidgets import (QWidget, QComboBox, QAction, qApp, QMainWindow, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QSizePolicy, QSlider)
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt
from video_widget import VideoWidget
from get_camera import Camera
from some_functions import toQImage, resizeImage
from main_widget import MainWidget

import sys
sys.path.append('expression_code/')
import main_expression

import scipy.misc
import numpy as np


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = "3DMM"
        self.setWindowIcon(QtGui.QIcon('images/1UP.ico'))
        self.setFixedSize(1200, 700)

        self.camera = Camera()
        self.video_widget = VideoWidget(self.camera)
        self.right_label = QLabel(self)
        self.main_widget = MainWidget(self.video_widget, self.right_label, self.camera)

        self.toolbar = self.addToolBar('Main Window')
        self.toolbar_emotions = self.addToolBar('Emotions')

        self.expression = "surprise"
        self.picture_taken = False
        self.portrait = None
        self.setCentralWidget(self.main_widget)
        self.initUI()

    def initUI(self):
        self.build_toolbar()
        # self.addToolBarBreak()

    def build_toolbar(self):

        slide_bar = QSlider(Qt.Horizontal)
        slide_bar.setMinimum(0)
        slide_bar.setMaximum(8)
        slide_bar.setValue(1)
        # slide_bar.setTickPosition(QSlider.TicksBelow)
        # slide_bar.setTickInterval(1)
        slide_bar.valueChanged.connect(self.camera.setZoom)
        self.toolbar.addWidget(slide_bar)

        take_photo = QAction(QIcon('images/get_photo.png'), 'Take_Picture', self)
        take_photo.setShortcut('Ctrl+Q')
        take_photo.triggered.connect(self.on_click)
        self.toolbar.addAction(take_photo)


        # build slide fo upload pre-built patterns.looking for the pattern class
        combo_box = QComboBox(self)
        combo_box.addItem("Surprise")
        combo_box.addItem("Happy")
        combo_box.addItem("Contempt")
        combo_box.addItem("Sadness")
        combo_box.addItem("Disgust")
        combo_box.addItem("Angry")
        combo_box.addItem("Fear")
        self.toolbar_emotions.addWidget(combo_box)
        combo_box.activated[str].connect(self.combo_changed)

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
        self.picture_taken = True
        self.portrait = self.camera.get_current_frame()
        self.portrait = resizeImage(self.portrait, 250)
        scipy.misc.imsave('expression_code/imgs/outfile.jpg', self.portrait)
        self.portrait = main_expression.apply_expression('expression_code/imgs/outfile.jpg',
                                                         expression=self.expression)
        self.portrait = resizeImage(self.portrait, 600)
        self.portrait = np.require(self.portrait, np.uint8, 'C')
        qim = toQImage(self.portrait)  # first convert to QImage
        qpix = QPixmap.fromImage(qim)  # then convert to QPixmap
        self.right_label.setPixmap(qpix)

    def combo_changed(self, text):
        self.expression = text.lower()
        if self.picture_taken:
            self.portrait = main_expression.apply_expression('expression_code/imgs/outfile.jpg',
                                                             expression=self.expression)
            self.portrait = resizeImage(self.portrait, 600)
            self.portrait = np.require(self.portrait, np.uint8, 'C')
            qim = toQImage(self.portrait)  # first convert to QImage
            qpix = QPixmap.fromImage(qim)  # then convert to QPixmap
            self.right_label.setPixmap(qpix)
        print(self.expression)
