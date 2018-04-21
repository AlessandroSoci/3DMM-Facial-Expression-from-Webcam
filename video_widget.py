from PyQt5.QtCore import (Qt, pyqtSignal)
from PyQt5.QtWidgets import (QWidget, QLabel, QSizePolicy)
from PyQt5.QtGui import QPixmap
from some_functions import toQImage
import numpy as np


class VideoWidget(QLabel):
    active = pyqtSignal()  # in order to work it has to be defined out of the constructor
    non_active = pyqtSignal()  # in order to work it has to be defined out of the constructor

    def __init__(self, camera):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.V_margin = 0
        self.H_margin = 0
        self.h = 0
        self.w = 0
        self.image = None
        self.camera = camera

        self.camera.updated.connect(self.new_image_slot, type=Qt.QueuedConnection)
        self.active.connect(self.camera.loop, type=Qt.QueuedConnection)
        self.non_active.connect(self.camera.deactivate, type=Qt.QueuedConnection)

    def activate(self):
        """Called upon activation of this view, emits the activated signal so that the Camera process can start"""
        self.active.emit()

    def deactivate(self):
        """Called upon deactivation of this view, emits the non_active signal so that the Camera process can stop"""
        self.non_active.emit()

    def new_image_slot(self):
        """Qt Slot for updated signal of the FaceRecogniser. Called every time a new frame is elaborated"""
        self.image = self.camera.get_current_frame()
        self.updateView()

    def set_model(self, image):
        self.image = image
        self.updateView()  # update the view to show the first frame

    def updateView(self):
        if self.image is None:
            return
        mat = self.image
        self.h = mat.shape[0]
        self.w = mat.shape[1]
        mat = np.require(mat, np.uint8, 'C')
        qim = toQImage(mat)  # first convert to QImage
        # size = QSize(self.h, self.w)
        qpix = QPixmap.fromImage(qim)  # then convert to QPixmap
        self.setPixmap(qpix)

