import numpy as np
import cv2
from video_widget import VideoWidget

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLineEdit, QLabel, QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout, QTableView, QSizePolicy, QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, pyqtSignal
from PyQt5.QtCore import Qt, QFileInfo, QUrl, QFile, QObject, pyqtSignal, QThread


class Camera(QThread):

    updated = pyqtSignal()  # in order to work it has to be defined out of the constructor
    person_identified = pyqtSignal()  # in order to work it has to be defined out of the constructor

    def __init__(self):
        super().__init__()

        self.currentFrame = None
        self.active = False
        self.zoom = 1

    def get_current_frame(self):
        """Getter for the currentFrame attribute"""
        return self.currentFrame

    def deactivate(self):
        """Method called to stop and deactivate the face recognition Thread"""
        self.active = False
        if self.isRunning():
            self.terminate()

    def loop(self):
        """Method called to initialize and start the face recognition Thread"""
        self.start()

    def run(self):
        """Main loop of this Thread"""
        self.active = True
        video_capture = cv2.VideoCapture(0)

        while self.active:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            if ret:
                frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
                end_y = frame.shape[1]
                end_x = frame.shape[0]
                diff = int(abs(end_y-end_x)/2)

                # creation of a square frame
                if end_y > end_x:
                    frame = frame[:, diff:end_y-diff, :]
                elif end_y < end_x:
                    frame = frame[diff:end_x-diff, :, :]

                # ZOOM
                frame_dimension = frame.shape[0]
                pre_zoom = self.zoom * frame_dimension
                if pre_zoom < frame_dimension:
                    pre_zoom = frame_dimension
                elif pre_zoom > 1.8*frame_dimension:
                    pre_zoom = 1.8*frame_dimension
                zoom = int((pre_zoom-frame_dimension)/2)
                frame = frame[zoom:frame_dimension-zoom, zoom:frame_dimension-zoom, :]

                frame = cv2.resize(frame, dsize=(600, 600), interpolation=cv2.INTER_CUBIC)

                # Store the current image
                self.currentFrame = frame
                self.updated.emit()
        cv2.release()

