from PyQt5.QtCore import (Qt, pyqtSignal)
from PyQt5.QtWidgets import (QWidget, QLabel, QSizePolicy)
from expression_code.main_expression import Model
from PyQt5.QtGui import QPixmap
from some_functions import toQImage, resizeImage
import numpy as np


class RightLabel(QLabel):

    active = pyqtSignal()  # in order to work it has to be defined out of the constructor
    non_active = pyqtSignal()  # in order to work it has to be defined out of the constructor

    def __init__(self, main_window, model):
        super().__init__(main_window)

        self.model = model
        self.image = None
        self.dictionary = None

        self.model.updated.connect(self.new_image, type=Qt.QueuedConnection)
        self.active.connect(self.model.start, type=Qt.QueuedConnection)
        self.non_active.connect(self.model.deactivate, type=Qt.QueuedConnection)

    def activate(self):
        self.active.emit()

    def deactivate(self):
        self.non_active.emit()

    # def wait_(self):
    def get_dictionary(self):
        return self.dictionary

    def set_model(self, model):
        self.model = model

    def new_image(self):
        self.image = self.model.get_image()
        self.dictionary = self.model.get_dictionary()
        self.update_view()

    def update_view(self):
        if self.image is None:
            return
        mat = resizeImage(self.image, 600)
        mat = np.require(mat, np.uint8, 'C')
        qim = toQImage(mat)  # first convert to QImage
        # size = QSize(self.h, self.w)
        qpix = QPixmap.fromImage(qim)  # then convert to QPixmap
        self.setPixmap(qpix)
        self.deactivate()
